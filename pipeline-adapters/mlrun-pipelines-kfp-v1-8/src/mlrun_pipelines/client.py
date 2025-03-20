# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import copy
import datetime
import logging
import os
import re
import tarfile
import tempfile
import time
import zipfile
from enum import IntEnum
from typing import Optional

import kfp_server_api
import kubernetes as k8s
import orjson
import yaml

import mlrun.utils
import mlrun_pipelines.common.client
import mlrun_pipelines.common.models

IN_CLUSTER_DNS_NAME = "ml-pipeline.{}.svc.cluster.local:8888"
KUBE_PROXY_PATH = "api/v1/namespaces/{}/services/ml-pipeline:http/proxy/"
KF_PIPELINES_SA_TOKEN_ENV = "KF_PIPELINES_SA_TOKEN_PATH"
KF_PIPELINES_SA_TOKEN_PATH = "/var/run/secrets/kubeflow/pipelines/token"
ROOT_PARAMETER_NAME = "pipeline-root"


class FilterOperations(IntEnum):
    UNKNOWN = 0
    EQUALS = 1
    NOT_EQUALS = 2
    GREATER_THAN = 3
    GREATER_THAN_EQUALS = 5
    LESS_THAN = 6
    LESS_THAN_EQUALS = 7


class ServiceAccountTokenVolumeCredentials:
    def __init__(
        self,
        path=None,
    ):
        self._token_path = (
            path or os.getenv(KF_PIPELINES_SA_TOKEN_ENV) or KF_PIPELINES_SA_TOKEN_PATH
        )

    def _get_token(self):
        try:
            with open(self._token_path) as f:
                token = f.read().strip()
            return token
        except OSError as e:
            logging.error(
                "Failed to read a token from file '%s' (%s).", self._token_path, str(e)
            )
            raise

    def refresh_api_key_hook(
        self,
        config: kfp_server_api.configuration,
    ):
        """Refresh the api key.

        This is a helper function for registering token refresh with swagger
        generated clients.

        Args:
            config (kubernetes.client.configuration.Configuration):
                The configuration object that the client uses.

                The Configuration object of the kubernetes client's is the same
                with kfp_server_api.configuration.Configuration.
        """
        config.api_key["authorization"] = self._get_token()


invalid_characters_regex = re.compile(r"[^-0-9a-z]+")
multiple_dashes_regex = re.compile(r"-+")


def sanitize_k8s_name(
    name: str,
):
    name = name.lower()
    cleaned_name = invalid_characters_regex.sub("-", name)
    cleaned_name = multiple_dashes_regex.sub("-", cleaned_name)
    return cleaned_name.lstrip("-").rstrip("-")


class Client(mlrun_pipelines.common.client.AbstractClient):
    def __init__(
        self,
        host: str = None,
        namespace: str = "mlrun",
    ):
        """Create a new instance of kfp client."""

        self._config = self._load_config(
            host=host,
            namespace=namespace,
        )
        self._api_client = kfp_server_api.api_client.ApiClient(
            configuration=self._config,
        )
        self._job_api = kfp_server_api.api.job_service_api.JobServiceApi(
            api_client=self._api_client,
        )
        self._run_api = kfp_server_api.api.run_service_api.RunServiceApi(
            api_client=self._api_client,
        )
        self._experiment_api = (
            kfp_server_api.api.experiment_service_api.ExperimentServiceApi(
                api_client=self._api_client,
            )
        )
        self._pipelines_api = (
            kfp_server_api.api.pipeline_service_api.PipelineServiceApi(
                api_client=self._api_client,
            )
        )
        self._upload_api = kfp_server_api.api.PipelineUploadServiceApi(
            api_client=self._api_client,
        )
        self._healthz_api = kfp_server_api.api.healthz_service_api.HealthzServiceApi(
            api_client=self._api_client,
        )

    def _get_config_with_default_credentials(
        self,
        config: kfp_server_api.configuration.Configuration,
    ):
        """Apply default credentials to the configuration object.

        This method accepts a Configuration object and extends it with
        some default credentials interface.
        """
        credentials = ServiceAccountTokenVolumeCredentials()
        config_copy = copy.deepcopy(config)

        try:
            credentials.refresh_api_key_hook(config_copy)
        except Exception:
            logging.warning(
                "Failed to set up default credentials. Proceeding without credentials..."
            )
            return config

        config.refresh_api_key_hook = credentials.refresh_api_key_hook
        config.api_key_prefix["authorization"] = "Bearer"
        return config

    def _load_config(
        self,
        host: str,
        namespace: str,
    ):
        config = kfp_server_api.configuration.Configuration()

        # Defaults to 'https' if host does not contain 'http' or 'https' protocol.
        if host and not host.startswith("http"):
            host = "https://" + host
        self._host = host or ""

        k8s.config.load_incluster_config()

        config.host = IN_CLUSTER_DNS_NAME.format(namespace)
        config = self._get_config_with_default_credentials(config)

        try:
            k8s.config.load_kube_config(
                client_configuration=config,
            )
        except Exception:
            logging.error("Failed to load kube config.")
            return config

        if config.host:
            config.host = config.host + "/" + KUBE_PROXY_PATH.format(namespace)
        return config

    def get_kfp_healthz(
        self,
        max_attempts: int = 5,
        interval_seconds: int = 5,
    ) -> Optional[kfp_server_api.ApiGetHealthzResponse]:
        count = 0
        response = None
        while not response:
            count += 1
            if count > max_attempts:
                raise TimeoutError(
                    f"Failed getting healthz endpoint after {max_attempts} attempts."
                )
            else:
                try:
                    response: kfp_server_api.ApiGetHealthzResponse = (
                        self._healthz_api.get_healthz()
                    )
                    return response
                except kfp_server_api.ApiException:
                    logging.exception(
                        f"Failed to get healthz info attempt {count} of {max_attempts}."
                    )
                    time.sleep(interval_seconds)

    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> kfp_server_api.ApiExperiment:
        """Create a new experiment.

        Args:
          name: The name of the experiment.
          description: Description of the experiment.
          namespace: Kubernetes namespace where the experiment should be created.


        Returns:
          An Experiment object.
        """
        experiment = None
        try:
            experiment = self.get_experiment(
                experiment_name=name,
                namespace=namespace,
            )
        except ValueError as error:
            if not str(error).startswith("No experiment is found with name"):
                raise error

        if not experiment:
            logging.info(f"Creating experiment {name}.")

            resource_references = []
            if namespace:
                key = kfp_server_api.models.ApiResourceKey(
                    id=namespace,
                    type=kfp_server_api.models.ApiResourceType.NAMESPACE,
                )
                reference = kfp_server_api.models.ApiResourceReference(
                    key=key,
                    relationship=kfp_server_api.models.ApiRelationship.OWNER,
                )
                resource_references.append(reference)

            experiment = kfp_server_api.models.ApiExperiment(
                name=name,
                description=description,
                resource_references=resource_references,
            )
            experiment = self._experiment_api.create_experiment(
                body=experiment,
            )
        return experiment

    def get_experiment(
        self,
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> kfp_server_api.ApiExperiment:
        if experiment_id is None and experiment_name is None:
            raise ValueError("Either experiment_id or experiment_name is required")
        if experiment_id is not None:
            return self._experiment_api.get_experiment(id=experiment_id)
        experiment_filter = orjson.dumps(
            {
                "predicates": [
                    {
                        "op": FilterOperations.EQUALS.value,
                        "key": "name",
                        "stringValue": experiment_name,
                    }
                ]
            }
        )
        if namespace:
            result = self._experiment_api.list_experiment(
                filter=experiment_filter,
                resource_reference_key_type=kfp_server_api.models.api_resource_type.ApiResourceType.NAMESPACE,
                resource_reference_key_id=namespace,
            )
        else:
            result = self._experiment_api.list_experiment(filter=experiment_filter)
        if not result.experiments:
            raise ValueError(f"No experiment is found with name {experiment_name}.")
        if len(result.experiments) > 1:
            raise ValueError(
                f"Multiple experiments is found with name {experiment_name}."
            )
        return result.experiments[0]

    def run_pipeline(
        self,
        experiment_id: str,
        job_name: str,
        pipeline_package_path: Optional[str] = None,
        params: Optional[dict] = None,
        pipeline_id: Optional[str] = None,
        version_id: Optional[str] = None,
        pipeline_root: Optional[str] = None,
        enable_caching: Optional[bool] = None,
        service_account: Optional[str] = None,
    ) -> kfp_server_api.ApiRun:
        if params is None:
            params = {}

        if pipeline_root is not None:
            params[ROOT_PARAMETER_NAME] = pipeline_root

        job_config = self._create_job_config(
            experiment_id=experiment_id,
            params=params,
            pipeline_package_path=pipeline_package_path,
            pipeline_id=pipeline_id,
            version_id=version_id,
            enable_caching=enable_caching,
        )
        run_body = kfp_server_api.models.ApiRun(
            pipeline_spec=job_config.spec,
            resource_references=job_config.resource_references,
            name=job_name,
            service_account=service_account,
        )

        response = self._run_api.create_run(
            body=run_body,
        )

        return response.run

    def list_runs(
        self,
        page_token: str = "",
        page_size: int = 10,
        sort_by: str = "",
        experiment_id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[str] = None,
    ) -> kfp_server_api.ApiListRunsResponse:
        if experiment_id is not None:
            response = self._run_api.list_runs(
                page_token=page_token,
                page_size=page_size,
                sort_by=sort_by,
                resource_reference_key_type=kfp_server_api.models.api_resource_type.ApiResourceType.EXPERIMENT,
                resource_reference_key_id=experiment_id,
                filter=filter,
            )
        elif namespace:
            response = self._run_api.list_runs(
                page_token=page_token,
                page_size=page_size,
                sort_by=sort_by,
                resource_reference_key_type=kfp_server_api.models.api_resource_type.ApiResourceType.NAMESPACE,
                resource_reference_key_id=namespace,
                filter=filter,
            )
        else:
            response = self._run_api.list_runs(
                page_token=page_token,
                page_size=page_size,
                sort_by=sort_by,
                filter=filter,
            )
        return response

    def get_run(self, run_id: str) -> kfp_server_api.ApiRun:
        return self._run_api.get_run(
            run_id=run_id,
        )

    def wait_for_run_completion(
        self,
        run_id: str,
        timeout: int,
        check_interval_seconds: int = 5,
    ) -> kfp_server_api.ApiRun:
        status = "Running:"
        start_time = datetime.datetime.now()
        if isinstance(timeout, datetime.timedelta):
            timeout = timeout.total_seconds()
        get_run_response = None

        while status not in mlrun_pipelines.common.modelsRunStatuses.stable_statuses():
            try:
                get_run_response = self._run_api.get_run(
                    run_id=run_id,
                )
            except kfp_server_api.ApiException as api_ex:
                raise api_ex
            status = get_run_response.run.status
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            logging.info("Waiting for the job to complete...")
            if elapsed_time > timeout:
                raise TimeoutError("Run timeout")
            time.sleep(check_interval_seconds)

        return get_run_response

    def upload_pipeline(
        self,
        pipeline_package_path: str,
        pipeline_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> kfp_server_api.ApiPipeline:
        response = self._upload_api.upload_pipeline(
            pipeline_package_path,
            name=pipeline_name,
            description=description,
        )
        return response

    @staticmethod
    def _normalize_retry_run(
        original_name: str,
        project: str,
    ) -> str:
        job_name = original_name.strip()
        proj_prefix = f"{project}-"
        retry_prefix = "Retry of "

        proj_prefix_len = len(proj_prefix)
        retry_prefix_len = len(retry_prefix)

        if job_name.startswith(proj_prefix):
            job_name = job_name[proj_prefix_len:].strip()
        if job_name.startswith(retry_prefix):
            job_name = job_name[retry_prefix_len:].strip()

        return f"{project}-Retry of {job_name}"

    def retry_run(
        self,
        run_id: str,
        project: str,
    ) -> str:
        """
        Retries a given run by its run ID. If the run is not in a valid state for retry,
        it creates a new run with the same pipeline and parameters.

        :param run_id: The ID of the run to retry.
        :type run_id: str
        :param project: The name of the project for the run.
        :type project: str
        :raises ApiException: If the API request fails during the retry or new run creation process.
        :raises ValueError: If the experiment ID cannot be found for the given run ID, or if
                            the original run does not contain a valid pipeline specification.
        :raises FileNotFoundError: If a temporary file for the workflow manifest cannot be created or accessed.
        :return: The ID of the new or retried run.
        :rtype: str
        """
        # Fetch run details
        run_details = self.get_run(run_id).run

        # Extract experiment ID from resource_references
        experiment_id = next(
            (
                ref.key.id
                for ref in run_details.resource_references
                if ref.key.type == "EXPERIMENT"
            ),
            None,
        )
        if not experiment_id:
            raise ValueError(f"Experiment ID not found for run ID: {run_id}")

        # If not retryable, create a new run
        pipeline_spec = run_details.pipeline_spec

        if not pipeline_spec.pipeline_id and not pipeline_spec.workflow_manifest:
            raise ValueError(
                "The original run does not contain a valid pipeline specification. "
                "Please ensure the pipeline has either a pipeline ID or workflow manifest."
            )

        workflow_manifest_path = None
        if not pipeline_spec.pipeline_id:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                delete=False,
            ) as temp_file:
                temp_file.write(pipeline_spec.workflow_manifest)
                workflow_manifest_path = temp_file.name

        # When retrying a KFP pipeline, we fetch the pipeline parameters from the previous run.
        # Due to an issue with the KFP server API, the pipeline parameters are returned as a list
        # containing a dictionary instead of a dictionary. We need to extract the dictionary from the list.
        pipeline_parameters = pipeline_spec.parameters
        if isinstance(pipeline_parameters, list):
            pipeline_parameters = pipeline_parameters[0]

        desired_prefix = f"{project}-Retry of "
        desired_prefix_lower = desired_prefix.lower()
        current_name = run_details.name.strip()

        if current_name.lower().startswith(desired_prefix_lower):
            job_name = current_name
        else:
            job_name = self._normalize_retry_run(
                current_name,
                project,
            )
        try:
            new_run = self.run_pipeline(
                experiment_id=experiment_id,
                job_name=job_name,
                pipeline_id=pipeline_spec.pipeline_id,
                params=pipeline_parameters,
                pipeline_package_path=workflow_manifest_path,
            )
            return new_run.id
        except kfp_server_api.OpenApiException as error:
            mlrun.utils.logger.error(
                "Could not trigger new run for run.",
                run_id=run_id,
                error=error,
            )
            raise error
        finally:
            if workflow_manifest_path and os.path.exists(workflow_manifest_path):
                os.remove(workflow_manifest_path)

    def _create_job_config(
        self,
        experiment_id: str,
        params: Optional[dict],
        pipeline_package_path: Optional[str],
        pipeline_id: Optional[str],
        version_id: Optional[str],
        enable_caching: Optional[bool],
    ):
        """Create a JobConfig with spec and resource_references.

        Args:
          experiment_id: The id of an experiment.
          pipeline_package_path: Local path of the pipeline package(the filename should end with one of the following .tar.gz, .tgz, .zip, .yaml, .yml).
          params: A dictionary with key (string) as param name and value (string) as param value.
          pipeline_id: The id of a pipeline.
          version_id: The id of a pipeline version.
            If both pipeline_id and version_id are specified, version_id will take precendence.
            If only pipeline_id is specified, the default version of this pipeline is used to create the run.
          enable_caching: Whether or not to enable caching for the run.
            This setting affects v2 compatible mode and v2 mode only.
            If not set, defaults to the compile time settings, which are True for all
            tasks by default, while users may specify different caching options for
            individual tasks.
            If set, the setting applies to all tasks in the pipeline -- overrides
            the compile time settings.

        Returns:
          A JobConfig object with attributes spec and resource_reference.
        """

        class JobConfig:
            def __init__(
                self,
                spec,
                resource_references,
            ):
                self.spec = spec
                self.resource_references = resource_references

        params = params or {}
        pipeline_orjson_string = None
        if pipeline_package_path:
            pipeline_obj = self._extract_pipeline_yaml(pipeline_package_path)

            # Caching option set at submission time overrides the compile time settings.
            if enable_caching is not None:
                self._override_caching_options(
                    pipeline_obj,
                    enable_caching,
                )

            pipeline_orjson_string = orjson.dumps(pipeline_obj)
        api_params = [
            kfp_server_api.ApiParameter(
                name=sanitize_k8s_name(name=k),
                value=str(v) if type(v) not in (list, dict) else orjson.dumps(v),
            )
            for k, v in params.items()
        ]
        resource_references = []
        key = kfp_server_api.models.ApiResourceKey(
            id=experiment_id,
            type=kfp_server_api.models.ApiResourceType.EXPERIMENT,
        )
        reference = kfp_server_api.models.ApiResourceReference(
            key=key,
            relationship=kfp_server_api.models.ApiRelationship.OWNER,
        )
        resource_references.append(reference)

        if version_id:
            key = kfp_server_api.models.ApiResourceKey(
                id=version_id,
                type=kfp_server_api.models.ApiResourceType.PIPELINE_VERSION,
            )
            reference = kfp_server_api.models.ApiResourceReference(
                key=key,
                relationship=kfp_server_api.models.ApiRelationship.CREATOR,
            )
            resource_references.append(reference)

        spec = kfp_server_api.models.ApiPipelineSpec(
            pipeline_id=pipeline_id,
            workflow_manifest=pipeline_orjson_string,
            parameters=api_params,
        )
        return JobConfig(
            spec=spec,
            resource_references=resource_references,
        )

    def _extract_pipeline_yaml(
        self,
        package_file: str,
    ):
        def _choose_pipeline_yaml_file(file_list: list[str]) -> str:
            yaml_files = [file for file in file_list if file.endswith(".yaml")]
            if len(yaml_files) == 0:
                raise ValueError(
                    "Invalid package. Missing pipeline yaml file in the package."
                )

            if "pipeline.yaml" in yaml_files:
                return "pipeline.yaml"
            else:
                if len(yaml_files) == 1:
                    return yaml_files[0]
                raise ValueError(
                    "Invalid package. There is no pipeline.yaml file and there are multiple yaml files."
                )

        if package_file.endswith(".tar.gz") or package_file.endswith(".tgz"):
            with tarfile.open(package_file, "r:gz") as tar:
                file_names = [member.name for member in tar if member.isfile()]
                pipeline_yaml_file = _choose_pipeline_yaml_file(file_names)
                with tar.extractfile(tar.getmember(pipeline_yaml_file)) as f:
                    return yaml.safe_load(f)
        elif package_file.endswith(".zip"):
            with zipfile.ZipFile(package_file, "r") as zip:
                pipeline_yaml_file = _choose_pipeline_yaml_file(zip.namelist())
                with zip.open(pipeline_yaml_file) as f:
                    return yaml.safe_load(f)
        elif package_file.endswith(".yaml") or package_file.endswith(".yml"):
            with open(package_file) as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(
                f"The package_file {package_file} should end with one of the following formats: [.tar.gz, .tgz, .zip, .yaml, .yml]"
            )

    def _override_caching_options(
        self,
        workflow: dict,
        enable_caching: bool,
    ):
        templates = workflow["spec"]["templates"]
        for template in templates:
            if (
                "metadata" in template
                and "labels" in template["metadata"]
                and "pipelines.kubeflow.org/enable_caching"
                in template["metadata"]["labels"]
            ):
                template["metadata"]["labels"][
                    "pipelines.kubeflow.org/enable_caching"
                ] = str(enable_caching).lower()
