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
from typing import Any, Optional

import kfp_server_api
import kubernetes as k8s
import orjson
import yaml

import mlrun_pipelines.common.client
import mlrun_pipelines.common.models
import mlrun_pipelines.models

IN_CLUSTER_DNS_NAME = "ml-pipeline.{}.svc.cluster.local:8888"
KUBE_PROXY_PATH = "api/v1/namespaces/{}/services/ml-pipeline:http/proxy/"
KF_PIPELINES_SA_TOKEN_ENV = "KF_PIPELINES_SA_TOKEN_PATH"
KF_PIPELINES_SA_TOKEN_PATH = "/var/run/secrets/kubeflow/pipelines/token"
ROOT_PARAMETER_NAME = "pipeline-root"

INVALID_CHARACTERS_REGEX = re.compile(r"[^-0-9a-z]+")
MULTIPLE_DASHES_REGEX = re.compile(r"-+")


class ServiceAccountTokenVolumeCredentials:
    def __init__(
        self,
        path: Optional[str] = None,
    ):
        self._token_path: str = (
            path or os.getenv(KF_PIPELINES_SA_TOKEN_ENV) or KF_PIPELINES_SA_TOKEN_PATH
        )

    def _read_token_from_file(
        self,
    ) -> Optional[str]:
        """
        Retrieve the token from the configured file path.

        :return: The token string if available, otherwise None.
        """
        try:
            with open(self._token_path) as f:
                token = f.read().strip()
            return token
        except FileNotFoundError:
            return None
        except OSError:
            raise ValueError("Failed to read service account token.")

    def refresh_api_key_hook(
        self,
        config: kfp_server_api.configuration.Configuration,
    ) -> None:
        """
        Refresh the API key in the provided configuration using the service account token.

        :param config: The configuration object used by the KFP client.
        """
        token = self._read_token_from_file()
        if token is not None:
            config.api_key["authorization"] = token


class JobConfig:
    """
    JobConfig encapsulates the pipeline spec and resource references needed to create or
    run a Kubeflow Pipelines job.
    """

    def __init__(
        self,
        pipeline_spec: kfp_server_api.models.ApiPipelineSpec,
        resource_references: list[kfp_server_api.models.ApiResourceReference],
    ):
        self.spec = pipeline_spec
        self.resource_references = resource_references


def sanitize_k8s_name(
    name: str,
) -> str:
    """
    Sanitize a Kubernetes resource name.

    This function converts the name to lowercase, replaces invalid characters with dashes,
    and removes any leading or trailing dashes.

    :param name: The original name to be sanitized.
    :return: A sanitized Kubernetes resource name.
    """
    max_k8s_name_length = 63
    name = name.lower()
    cleaned_name = INVALID_CHARACTERS_REGEX.sub("-", name)
    cleaned_name = MULTIPLE_DASHES_REGEX.sub("-", cleaned_name)
    cleaned_name = cleaned_name.lstrip("-").rstrip("-")
    if len(cleaned_name) > max_k8s_name_length:
        raise ValueError(
            f"Kubernetes resource name '{cleaned_name}' is too long. "
            f"Max length is {max_k8s_name_length} characters."
        )
    return cleaned_name


class Client(
    mlrun_pipelines.common.client.AbstractClient,
):
    def __init__(
        self,
        host: Optional[str] = None,
        namespace: str = "mlrun",
    ):
        self._config: kfp_server_api.configuration.Configuration = self._load_config(
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

    @staticmethod
    def _get_config_with_default_credentials(
        config: kfp_server_api.configuration.Configuration,
    ) -> kfp_server_api.configuration.Configuration:
        """
        Apply default credentials to the KFP configuration.

        This method updates the provided KFP configuration with a service account token if possible.

        :param config: The original KFP configuration.
        :return: The updated configuration with default credentials.
        """
        credentials = ServiceAccountTokenVolumeCredentials()
        config_copy = copy.deepcopy(config)

        try:
            credentials.refresh_api_key_hook(config_copy)
        except Exception:
            logging.warning(
                "Failed to set up credentials. Proceeding without credentials..."
            )
            return config

        config.refresh_api_key_hook = credentials.refresh_api_key_hook
        config.api_key_prefix["authorization"] = "Bearer"
        return config

    def _load_config(
        self,
        host: Optional[str],
        namespace: str,
    ) -> kfp_server_api.configuration.Configuration:
        """
        Load and configure Kubernetes settings for the KFP client.

        This method loads in-cluster configuration, applies default credentials,
        and attempts to load kubeconfig for the given namespace.

        :param host:      An optional host URL for the KFP API.
        :param namespace: The Kubernetes namespace for pipeline resources.
        :return: A fully configured kfp_server_api.configuration.Configuration object.
        """
        config = kfp_server_api.configuration.Configuration()

        # If host is provided without http/https, prepend https://
        if host and not host.startswith("http"):
            host = "https://" + host
        self._host: str = host or ""

        k8s.config.load_incluster_config()

        config.host = IN_CLUSTER_DNS_NAME.format(namespace)
        config = self._get_config_with_default_credentials(config)

        try:
            k8s.config.load_kube_config(
                client_configuration=config,
            )
        except Exception:
            logging.info("Failed to load kube config.")
            return config

        if config.host:
            config.host += "/" + KUBE_PROXY_PATH.format(namespace)
        return config

    def get_kfp_healthz(
        self,
        max_attempts: int = 5,
        interval_seconds: int = 5,
    ) -> Optional[kfp_server_api.ApiGetHealthzResponse]:
        """
        Retrieve the healthz status of the KFP API.

        This method retries multiple times until the endpoint responds or a timeout occurs.

        :param max_attempts:     Maximum number of retry attempts.
        :param interval_seconds: Interval (in seconds) between attempts.
        :return: A valid ApiGetHealthzResponse if successful, otherwise None.
        :raises TimeoutError: If the endpoint is not reachable after the specified retries.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                return self._healthz_api.get_healthz()
            except kfp_server_api.ApiException:
                logging.exception(
                    "Failed to retrieve KFP healthz info on attempt %d of %d.",
                    attempt,
                    max_attempts,
                )
                time.sleep(interval_seconds)
        raise TimeoutError(
            f"Failed to get healthz endpoint after {max_attempts} attempts."
        )

    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> kfp_server_api.ApiExperiment:
        """
        Create a new experiment if it does not already exist.

        This method searches for an experiment by name (and optional namespace), and creates
        it if not found. If found, the existing experiment is returned.

        :param name:         The name of the experiment to create or retrieve.
        :param description:  A description for the experiment.
        :param namespace:    An optional Kubernetes namespace.
        :return: An ApiExperiment object representing the experiment.
        :raises ValueError:  If multiple experiments with the same name are found.
        """
        experiment: Optional[kfp_server_api.ApiExperiment] = None
        try:
            experiment = self.get_experiment(
                experiment_name=name,
                namespace=namespace,
            )
        except ValueError as error:
            if not str(error).startswith("No experiment is found with name"):
                raise error

        if not experiment:
            logging.info("Creating experiment '%s'.", name)
            resource_references: list[kfp_server_api.models.ApiResourceReference] = []
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
        """
        Retrieve an experiment by ID or name.

        This method fetches an experiment using its unique ID or by searching for the name
        (optionally in a specified namespace).

        :param experiment_id:   The ID of the experiment to retrieve.
        :param experiment_name: The name of the experiment to retrieve.
        :param namespace:       An optional Kubernetes namespace for filtering by name.
        :return: An ApiExperiment object representing the experiment.
        :raises ValueError: If neither experiment_id nor experiment_name is provided, or if
                            multiple experiments share the same name.
        """
        if experiment_id is None and experiment_name is None:
            raise ValueError("Either experiment_id or experiment_name is required")

        if experiment_id is not None:
            return self._experiment_api.get_experiment(id=experiment_id)

        filter_json = orjson.dumps(
            {
                "predicates": [
                    {
                        "op": mlrun_pipelines.models.FilterOperations.EQUALS.value,
                        "key": "name",
                        "stringValue": experiment_name,
                    }
                ]
            }
        ).decode()

        if namespace:
            result = self._experiment_api.list_experiment(
                filter=filter_json,
                resource_reference_key_type=(
                    kfp_server_api.models.api_resource_type.ApiResourceType.NAMESPACE
                ),
                resource_reference_key_id=namespace,
            )
        else:
            result = self._experiment_api.list_experiment(filter=filter_json)

        if not result.experiments:
            raise ValueError(f"No experiment is found with name {experiment_name}.")
        if len(result.experiments) > 1:
            raise ValueError(f"Multiple experiments found with name {experiment_name}.")
        return result.experiments[0]

    def run_pipeline(
        self,
        experiment_id: str,
        job_name: str,
        pipeline_package_path: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
        pipeline_id: Optional[str] = None,
        version_id: Optional[str] = None,
        pipeline_root: Optional[str] = None,
        enable_caching: Optional[bool] = None,
        service_account: Optional[str] = None,
    ) -> kfp_server_api.ApiRun:
        """
        Run a pipeline within a specified experiment.

        This method submits a pipeline run using various optional arguments like a pipeline
        package path, pipeline ID, parameters, and caching settings.

        :param experiment_id:         The ID of the experiment to run the pipeline in.
        :param job_name:              The name to assign to this pipeline run.
        :param pipeline_package_path: An optional path to the pipeline package file (tar.gz, zip, yaml).
        :param params:                An optional dictionary of pipeline parameters.
        :param pipeline_id:           An optional pipeline ID. If provided, the client uses the existing pipeline.
        :param version_id:            An optional pipeline version ID.
        :param pipeline_root:         An optional root path for pipeline outputs.
        :param enable_caching:        A flag to enable or disable pipeline caching.
        :param service_account:       An optional Kubernetes service account to run the pipeline.
        :return: An ApiRun object representing the created pipeline run.
        """
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
        response = self._run_api.create_run(body=run_body)
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
        """
        List pipeline runs with optional filters.

        This method retrieves runs, optionally filtering by experiment ID, namespace, or custom filters.
        Pagination and sorting are also supported.

        :param page_token:    A token for pagination.
        :param page_size:     Number of runs to retrieve per request.
        :param sort_by:       A string specifying how to sort the results.
        :param experiment_id: An optional experiment ID to filter runs by.
        :param namespace:     An optional namespace to filter runs by.
        :param filter:       A custom filter string (if any).
        :return: An ApiListRunsResponse object containing the runs.
        """
        if experiment_id is not None:
            response = self._run_api.list_runs(
                page_token=page_token,
                page_size=page_size,
                sort_by=sort_by,
                resource_reference_key_type=(
                    kfp_server_api.models.api_resource_type.ApiResourceType.EXPERIMENT
                ),
                resource_reference_key_id=experiment_id,
                filter=filter,
            )
        elif namespace:
            response = self._run_api.list_runs(
                page_token=page_token,
                page_size=page_size,
                sort_by=sort_by,
                resource_reference_key_type=(
                    kfp_server_api.models.api_resource_type.ApiResourceType.NAMESPACE
                ),
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

    def get_run(
        self,
        run_id: str,
    ) -> kfp_server_api.ApiRunDetail:
        """
        Retrieve details of a specific pipeline run.

        :param run_id: The unique ID of the run to retrieve.
        :return: An ApiRun object with the run details.
        """
        logging.info("Getting run details for run ID: %s", run_id)
        return self._run_api.get_run(
            run_id=run_id,
        )

    def wait_for_run_completion(
        self,
        run_id: str,
        timeout: int,
        check_interval_seconds: int = 5,
    ) -> kfp_server_api.ApiRun:
        """
        Wait for a pipeline run to reach a stable status (e.g., Succeeded or Failed).

        This method polls the run status at a specified interval until it completes
        or a timeout is reached.

        :param run_id:               The unique ID of the run.
        :param timeout:              The total time in seconds to wait for completion.
        :param check_interval_seconds: How often (in seconds) to poll the run status.
        :return: An ApiRun object describing the run at final status.
        :raises TimeoutError: If the run does not complete before the timeout.
        """
        status: str = "Running:"
        start_time: datetime.datetime = datetime.datetime.now()
        if isinstance(timeout, datetime.timedelta):
            timeout = int(timeout.total_seconds())
        get_run_response: Optional[kfp_server_api.ApiRun] = None

        while status not in mlrun_pipelines.common.models.RunStatuses.stable_statuses():
            try:
                get_run_response: kfp_server_api.ApiRunDetail = self._run_api.get_run(
                    run_id=run_id
                )
            except kfp_server_api.ApiException as api_ex:
                raise api_ex
            status = get_run_response.run.status
            elapsed_time: float = (datetime.datetime.now() - start_time).total_seconds()
            logging.info("Waiting for the job to complete (status: %s)...", status)
            if elapsed_time > timeout:
                raise TimeoutError(
                    f"Run {run_id} did not complete within {timeout} seconds."
                )
            time.sleep(check_interval_seconds)

        return get_run_response

    def upload_pipeline(
        self,
        pipeline_package_path: str,
        pipeline_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> kfp_server_api.ApiPipeline:
        """
        Upload a pipeline package file to Kubeflow Pipelines.

        :param pipeline_package_path: Path to the pipeline package file (zip, tar.gz, yaml, etc.).
        :param pipeline_name:         An optional name to assign to the pipeline.
        :param description:           An optional description for the pipeline.
        :return: An ApiPipeline object representing the uploaded pipeline.
        """
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
        """
        Normalize a job name for retry attempts.

        This method ensures the new job name references the project and indicates that
        it is a retry of the original run.

        :param original_name: The original pipeline run name.
        :param project:       The project name or prefix to include.
        :return: A standardized retry name (e.g., "myproject-Retry of original_name").
        """
        job_name: str = original_name.strip()
        proj_prefix: str = f"{project}-"
        retry_prefix: str = "Retry of "

        if job_name.startswith(proj_prefix):
            job_name = job_name[len(proj_prefix) :].strip()
        if job_name.startswith(retry_prefix):
            job_name = job_name[len(retry_prefix) :].strip()

        return f"{project}-Retry of {job_name}"

    def retry_run(
        self,
        run_id: str,
        project: str,
    ) -> Optional[str]:
        """
        Retry a previous run by ID, or create a new run with the same pipeline and parameters.

        This method attempts to reuse the pipeline specification and parameters from the
        original run. If the original run is not in a retryable state (e.g. lacks pipeline
        spec), it creates a fresh run.

        :param run_id:  The ID of the run to be retried.
        :param project: The name of the project this run belongs to.
        :return: The ID of the new or retried run if successful, otherwise None.
        :raises ValueError: If the experiment ID or pipeline spec cannot be found.
        :raises kfp_server_api.ApiException: If the creation of the new run fails.
        """
        existing_run_details = self.get_run(run_id).run
        experiment_id: Optional[str] = next(
            (
                ref.key.id
                for ref in existing_run_details.resource_references
                if ref.key.type == "EXPERIMENT"
            ),
            None,
        )
        if not experiment_id:
            raise ValueError(f"Experiment ID not found for run ID: {run_id}")

        pipeline_spec = existing_run_details.pipeline_spec
        if not pipeline_spec.pipeline_id and not pipeline_spec.workflow_manifest:
            raise ValueError(
                "The original run does not contain a valid pipeline specification. "
                "Please ensure the pipeline has either a pipeline ID or workflow manifest."
            )

        # Extract workflow manifest, if no pipeline_id is available
        workflow_manifest_path: Optional[str] = None
        if not pipeline_spec.pipeline_id:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                delete=False,
            ) as temp_file:
                temp_file.write(pipeline_spec.workflow_manifest)
                workflow_manifest_path = temp_file.name

        # KFP server API may return pipeline parameters as a list containing a single dict
        pipeline_parameters: Any = pipeline_spec.parameters
        if isinstance(pipeline_parameters, list) and pipeline_parameters:
            pipeline_parameters = pipeline_parameters[0]

        current_name: str = existing_run_details.name.strip()
        desired_prefix: str = f"{project}-Retry of "
        if not current_name.lower().startswith(desired_prefix.lower()):
            job_name = self._normalize_retry_run(
                original_name=current_name,
                project=project,
            )
        else:
            job_name = current_name

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
            logging.error(
                "Could not trigger new run for run %s, error: %s",
                run_id,
                error,
            )
            raise error
        finally:
            if workflow_manifest_path and os.path.exists(workflow_manifest_path):
                os.remove(workflow_manifest_path)

    def _create_job_config(
        self,
        experiment_id: str,
        params: Optional[dict[str, Any]],
        pipeline_package_path: Optional[str],
        pipeline_id: Optional[str],
        version_id: Optional[str],
        enable_caching: Optional[bool],
    ) -> JobConfig:
        """
        Create a JobConfig object holding the pipeline spec and resource references.

        This method handles assembling the pipeline spec from a package (or existing ID)
        and optionally applies caching overrides.

        :param experiment_id:         The experiment ID to which this run will be associated.
        :param params:                A dictionary of pipeline parameters.
        :param pipeline_package_path: An optional path to a pipeline package file.
        :param pipeline_id:           An optional existing pipeline ID.
        :param version_id:            An optional pipeline version ID (takes precedence if provided).
        :param enable_caching:        Optional boolean to enable or disable caching.
        :return: A fully configured JobConfig instance.
        """
        params = params or {}
        pipeline_json_string: Optional[str] = None

        if pipeline_package_path:
            pipeline_obj = self._parse_pipeline_obj(
                package_file=pipeline_package_path,
            )
            if enable_caching is not None:
                self._override_caching_options(
                    workflow=pipeline_obj,
                    enable_caching=enable_caching,
                )
            pipeline_json_string = orjson.dumps(pipeline_obj).decode()

        api_params: list[kfp_server_api.ApiParameter] = [
            kfp_server_api.ApiParameter(
                name=sanitize_k8s_name(key),
                value=(
                    str(value)
                    if not isinstance(value, (list, dict))
                    else orjson.dumps(value)
                ),
            )
            for key, value in params.items()
        ]

        resource_references = [
            kfp_server_api.models.ApiResourceReference(
                key=kfp_server_api.models.ApiResourceKey(
                    id=experiment_id,
                    type=kfp_server_api.models.ApiResourceType.EXPERIMENT,
                ),
                relationship=kfp_server_api.models.ApiRelationship.OWNER,
            )
        ]

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
            workflow_manifest=pipeline_json_string,
            parameters=api_params,
        )

        return JobConfig(
            pipeline_spec=spec,
            resource_references=resource_references,
        )

    @staticmethod
    def _parse_pipeline_obj(
        package_file: str,
    ) -> Any:
        """
        Extract the pipeline YAML from a package file.

        This method supports the following file formats: .tar.gz, .tgz, .zip, .yaml, .yml.
        It returns a parsed YAML object representing the pipeline definition.

        :param package_file: Path to the pipeline package file.
        :return: Parsed YAML content of the pipeline definition.
        :raises ValueError: If the package is invalid or missing a pipeline.yaml file.
        """

        def _choose_pipeline_yaml_file(
            file_list: list[str],
        ) -> str:
            pipeline_file_name = "pipeline.yaml"
            yaml_files: list[str] = [
                file for file in file_list if file.endswith(".yaml")
            ]
            if not yaml_files:
                raise ValueError(
                    "Invalid package. Missing pipeline yaml file in the package."
                )
            if pipeline_file_name in yaml_files:
                return pipeline_file_name
            if len(yaml_files) == 1:
                return yaml_files[0]
            raise ValueError(
                "Invalid package. Multiple YAML files found without a 'pipeline.yaml'."
            )

        if package_file.endswith(".tar.gz") or package_file.endswith(".tgz"):
            with tarfile.open(package_file, "r:gz") as tar:
                file_names = [member.name for member in tar if member.isfile()]
                pipeline_yaml_file = _choose_pipeline_yaml_file(file_names)
                with tar.extractfile(tar.getmember(pipeline_yaml_file)) as f:
                    return yaml.safe_load(f)

        elif package_file.endswith(".zip"):
            with zipfile.ZipFile(package_file, "r") as zip_file:
                pipeline_yaml_file = _choose_pipeline_yaml_file(zip_file.namelist())
                with zip_file.open(pipeline_yaml_file) as f:
                    return yaml.safe_load(f)

        elif package_file.endswith(".yaml") or package_file.endswith(".yml"):
            with open(package_file) as f:
                return yaml.safe_load(f)

        else:
            raise ValueError(
                f"The package_file '{package_file}' should end with one of "
                f"the following formats: [.tar.gz, .tgz, .zip, .yaml, .yml]"
            )

    @staticmethod
    def _override_caching_options(
        workflow: dict[str, Any],
        enable_caching: bool,
    ) -> None:
        """
        Override caching behavior in a pipeline workflow manifest.

        This method sets a label on each template in the Argo workflow, controlling
        whether pipeline steps use caching.

        :param workflow:       A dictionary representing the pipeline's Argo workflow.
        :param enable_caching: A boolean indicating whether caching should be enabled.
        """
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
