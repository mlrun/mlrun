import copy
import datetime
import logging
import os
import re
import tarfile
import tempfile
import time
import zipfile
from typing import Any, Optional, TextIO

import kfp_server_api
import kubernetes as k8s
import orjson
import yaml

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
        try:
            with open(self._token_path) as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
        except OSError:
            raise ValueError("Failed to read service account token.")

    def refresh_api_key_hook(
        self,
        config: kfp_server_api.configuration.Configuration,
    ) -> None:
        token = self._read_token_from_file()
        if token is not None:
            config.api_key["authorization"] = token


def sanitize_k8s_name(name: str) -> str:
    name = name.lower()
    cleaned_name = INVALID_CHARACTERS_REGEX.sub("-", name)
    cleaned_name = MULTIPLE_DASHES_REGEX.sub("-", cleaned_name)
    return cleaned_name.strip("-")


class Client:
    def __init__(
        self,
        host: Optional[str] = None,
        namespace: str = "mlrun",
    ):
        self._config = self._load_config(host, namespace)
        self._api_client = kfp_server_api.api_client.ApiClient(self._config)
        self._run_api = kfp_server_api.api.run_service_api.RunServiceApi(
            self._api_client
        )
        self._experiment_api = (
            kfp_server_api.api.experiment_service_api.ExperimentServiceApi(
                self._api_client
            )
        )
        self._pipelines_api = (
            kfp_server_api.api.pipeline_service_api.PipelineServiceApi(self._api_client)
        )
        self._upload_api = kfp_server_api.api.PipelineUploadServiceApi(self._api_client)
        self._healthz_api = kfp_server_api.api.healthz_service_api.HealthzServiceApi(
            self._api_client
        )

    @staticmethod
    def _get_config_with_default_credentials(
        config: kfp_server_api.configuration.Configuration,
    ) -> kfp_server_api.configuration.Configuration:
        credentials = ServiceAccountTokenVolumeCredentials()
        config_copy = copy.deepcopy(config)
        try:
            credentials.refresh_api_key_hook(config_copy)
        except Exception:
            logging.warning("Proceeding without credentials...")
            return config
        config.refresh_api_key_hook = credentials.refresh_api_key_hook
        config.api_key_prefix["authorization"] = "Bearer"
        return config

    def _load_config(
        self,
        host: Optional[str],
        namespace: str,
    ) -> kfp_server_api.configuration.Configuration:
        config = kfp_server_api.configuration.Configuration()
        if host and not host.startswith("http"):
            host = "https://" + host
        self._host: str = host or ""
        try:
            k8s.config.load_incluster_config()
            config.host = IN_CLUSTER_DNS_NAME.format(namespace)
            config = self._get_config_with_default_credentials(config)
            return config
        except Exception:
            pass

        try:
            k8s.config.load_kube_config(client_configuration=config)
            if config.host:
                config.host += "/" + KUBE_PROXY_PATH.format(namespace)
        except Exception:
            logging.info("Failed to load kube config.")
        return self._get_config_with_default_credentials(config)

    def get_kfp_healthz(
        self,
        max_attempts: int = 5,
        interval_seconds: int = 5,
    ) -> Optional[kfp_server_api.models.V2beta1GetHealthzResponse]:
        count = 0
        while count < max_attempts:
            count += 1
            try:
                return self._healthz_api.healthz_service_get_healthz()
            except kfp_server_api.ApiException:
                logging.exception("Attempt %d of %d failed.", count, max_attempts)
                time.sleep(interval_seconds)
        raise TimeoutError("Could not get KFP health after retries.")

    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> kfp_server_api.models.V2beta1Experiment:
        try:
            exp = self.get_experiment(experiment_name=name, namespace=namespace)
            return exp
        except ValueError as e:
            if not str(e).startswith("No experiment is found with name"):
                raise

        body = kfp_server_api.models.V2beta1Experiment(
            display_name=name,
            description=description,
        )
        return self._experiment_api.experiment_service_create_experiment(
            body=body,
            namespace=namespace,
        )

    def get_experiment(
        self,
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> kfp_server_api.models.V2beta1Experiment:
        if not experiment_id and not experiment_name:
            raise ValueError("Either experiment_id or experiment_name is required")

        if experiment_id:
            return self._experiment_api.experiment_service_get_experiment(
                experiment_id=experiment_id,
            )

        filter_json = orjson.dumps(
            {
                "predicates": [
                    {
                        "op": "EQUALS",
                        "key": "name",
                        "stringValue": experiment_name,
                    }
                ]
            }
        ).decode()

        if namespace:
            result = self._experiment_api.experiment_service_list_experiments(
                filter=filter_json,
                namespace=namespace,
            )
        else:
            result = self._experiment_api.experiment_service_list_experiments(
                filter=filter_json
            )

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
        cache_key: Optional[str] = None,
        service_account: Optional[str] = None,
    ) -> kfp_server_api.models.V2beta1Run:
        if not params:
            params = {}
        if pipeline_root is not None:
            params[ROOT_PARAMETER_NAME] = pipeline_root

        job_config = self._create_job_config(
            params=params,
            pipeline_package_path=pipeline_package_path,
            pipeline_id=pipeline_id,
            version_id=version_id,
            enable_caching=enable_caching,
            cache_key=cache_key,
            pipeline_root=pipeline_root,
        )

        run_body = kfp_server_api.V2beta1Run(
            experiment_id=experiment_id,
            display_name=job_name,
            pipeline_spec=job_config.pipeline_spec,
            pipeline_version_reference=job_config.pipeline_version_reference,
            runtime_config=job_config.runtime_config,
            service_account=service_account,
        )
        return self._run_api.run_service_create_run(body=run_body)

    def list_runs(
        self,
        page_token: str = "",
        page_size: int = 10,
        sort_by: str = "",
        experiment_id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[str] = None,
    ) -> kfp_server_api.V2beta1ListRunsResponse:
        """List runs.

        Args:
            page_token: Page token for obtaining page from paginated response.
            page_size: Size of the page.
            sort_by: Sort string of format ``'[field_name]', '[field_name] desc'``. For example, ``'display_name desc'``.
            experiment_id: Experiment ID to filter upon
            namespace: Kubernetes namespace to use. Used for multi-user deployments. For single-user deployments, this should be left as ``None``.
            filter: A url-encoded, JSON-serialized Filter protocol buffer
             Example:
                    json.dumps(
                        {
                            "predicates": [
                                {
                                    "operation": "EQUALS",
                                    "key": "display_name",
                                    "stringValue": "my-name",
                                }
                            ]
                        }
                    )

          Returns:
            ``V2beta1ListRunsResponse`` object.
        """
        if experiment_id is not None:
            return self._run_api.run_service_list_runs(
                page_token=page_token,
                page_size=page_size,
                sort_by=sort_by,
                experiment_id=experiment_id,
                filter=filter,
            )

        elif namespace is not None:
            return self._run_api.run_service_list_runs(
                page_token=page_token,
                page_size=page_size,
                sort_by=sort_by,
                namespace=namespace,
                filter=filter,
            )

        else:
            return self._run_api.run_service_list_runs(
                page_token=page_token,
                page_size=page_size,
                sort_by=sort_by,
                filter=filter,
            )

    def get_run(self, run_id: str) -> kfp_server_api.models.V2beta1Run:
        return self._run_api.run_service_get_run(run_id=run_id)

    def wait_for_run_completion(
        self,
        run_id: str,
        timeout: int,
        check_interval_seconds: int = 5,
    ) -> kfp_server_api.models.V2beta1Run:
        start_time = datetime.datetime.now()
        while True:
            run_detail = self.get_run(run_id).run_details
            status = run_detail.run.status
            if status not in ("Running", "Pending", None):
                return run_detail.run
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                raise TimeoutError(f"Run {run_id} timed out after {timeout} seconds.")
            time.sleep(check_interval_seconds)

    def upload_pipeline(
        self,
        pipeline_package_path: str,
        pipeline_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> kfp_server_api.models.V2beta1Pipeline:
        return self._upload_api.upload_pipeline(
            pipeline_package_path,
            name=pipeline_name,
            description=description,
        )

    def retry_run(
        self,
        run_id: str,
        project: str,
    ) -> Optional[str]:
        existing_run = self.get_run(run_id)
        experiment_id = existing_run.experiment_id
        if not experiment_id:
            raise ValueError("Cannot find experiment ID to retry run.")

        pipeline_spec = existing_run["pipeline_spec"]
        if not (pipeline_spec["pipeline_id"] or pipeline_spec["workflow_manifest"]):
            raise ValueError(
                "The original run does not have a pipeline_id or workflow_manifest."
            )

        workflow_manifest_path = None
        if not pipeline_spec.pipeline_id:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as temp_file:
                temp_file.write(pipeline_spec.workflow_manifest)
                workflow_manifest_path = temp_file.name

        parameters = {}
        if pipeline_spec.parameters:
            if isinstance(pipeline_spec.parameters, list):
                for p in pipeline_spec.parameters:
                    parameters[p.name] = p.value
            else:
                parameters = pipeline_spec.parameters

        original_name = existing_run.name.strip()
        prefix = f"{project}-Retry of "
        if not original_name.startswith(prefix):
            new_name = prefix + original_name
        else:
            new_name = original_name

        try:
            new_run = self.run_pipeline(
                experiment_id=experiment_id,
                job_name=new_name,
                pipeline_id=pipeline_spec.pipeline_id,
                pipeline_package_path=workflow_manifest_path,
                params=parameters,
            )
            return new_run.id
        finally:
            if workflow_manifest_path and os.path.exists(workflow_manifest_path):
                os.remove(workflow_manifest_path)

    @staticmethod
    def _create_job_config(
        params: Optional[dict[str, Any]],
        pipeline_package_path: Optional[str],
        pipeline_id: Optional[str],
        version_id: Optional[str],
        enable_caching: Optional[bool],
        cache_key: Optional[str],
        pipeline_root: Optional[str],
    ) -> mlrun_pipelines.models.JobConfig:
        from_spec = pipeline_package_path is not None
        from_template = pipeline_id is not None or version_id is not None
        if from_spec == from_template:
            raise ValueError(
                "Must specify either `pipeline_pacakge_path` or both `pipeline_id` and `version_id`."
            )
        if (pipeline_id is None) != (version_id is None):
            raise ValueError(
                "To run a pipeline from an existing template, both `pipeline_id` and `version_id` are required."
            )

        if params is None:
            params = {}

        pipeline_spec = None
        if pipeline_package_path:
            pipeline_doc = _extract_pipeline_yaml(pipeline_package_path)

            if enable_caching is not None:
                _override_caching_options(
                    pipeline_doc.pipeline_spec, enable_caching, cache_key
                )
            pipeline_spec = pipeline_doc.to_dict()

        pipeline_version_reference = None
        if pipeline_id is not None and version_id is not None:
            pipeline_version_reference = kfp_server_api.V2beta1PipelineVersionReference(
                pipeline_id=pipeline_id, pipeline_version_id=version_id
            )

        runtime_config = kfp_server_api.V2beta1RuntimeConfig(
            pipeline_root=pipeline_root,
            parameters=params,
        )
        return mlrun_pipelines.models._JobConfig(
            pipeline_spec=pipeline_spec,
            pipeline_version_reference=pipeline_version_reference,
            runtime_config=runtime_config,
        )


def _extract_pipeline_yaml(package_file: str) -> mlrun_pipelines.models.PipelineDoc:
    def _choose_pipeline_file(file_list: list[str]) -> str:
        pipeline_files = [file for file in file_list if file.endswith(".yaml")]
        if not pipeline_files:
            raise ValueError(
                "Invalid package. Missing pipeline yaml file in the package."
            )

        if "pipeline.yaml" in pipeline_files:
            return "pipeline.yaml"
        elif len(pipeline_files) == 1:
            return pipeline_files[0]
        else:
            raise ValueError(
                "Invalid package. There is no pipeline.json file or there "
                "are multiple yaml files."
            )

    def _safe_load_yaml(stream: TextIO) -> mlrun_pipelines.models.PipelineDoc:
        docs = yaml.safe_load_all(stream)
        pipeline_spec_dict = None
        platform_spec_dict = {}
        for doc in docs:
            if pipeline_spec_dict is None:
                pipeline_spec_dict = doc
            else:
                platform_spec_dict.update(doc)

        return mlrun_pipelines.models.PipelineDoc(
            pipeline_spec=mlrun_pipelines.models.PipelineSpec(**pipeline_spec_dict),
            platform_spec=mlrun_pipelines.models.PlatformSpec(**platform_spec_dict),
        )

    if package_file.endswith(".tar.gz") or package_file.endswith(".tgz"):
        with tarfile.open(package_file, "r:gz") as tar:
            file_names = [member.name for member in tar if member.isfile()]
            pipeline_file = _choose_pipeline_file(file_names)
            with tar.extractfile(tar.getmember(pipeline_file)) as f:
                return _safe_load_yaml(f)
    elif package_file.endswith(".zip"):
        with zipfile.ZipFile(package_file, "r") as zip:
            pipeline_file = _choose_pipeline_file(zip.namelist())
            with zip.open(pipeline_file) as f:
                return _safe_load_yaml(f)
    elif package_file.endswith(".yaml") or package_file.endswith(".yml"):
        with open(package_file) as f:
            return _safe_load_yaml(f)
    else:
        raise ValueError(
            f"The package_file {package_file} should end with one of the "
            "following formats: [.tar.gz, .tgz, .zip, .yaml, .yml]."
        )


def _override_caching_options(
    pipeline_spec: mlrun_pipelines.models.PipelineSpec,
    enable_caching: bool,
    cache_key: Optional[str] = None,
) -> None:
    """Overrides caching options.

    Args:
        pipeline_spec: The PipelineSpec object to update in-place.
        enable_caching: Overrides options, one of True, False.
        cache_key: Overrides cache_key, default None, no-op.
    """
    for _, task_spec in pipeline_spec.root.dag.tasks.items():
        task_spec.caching_options.enable_cache = enable_caching
        if cache_key:
            task_spec.caching_options.cache_key = cache_key
