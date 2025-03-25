import copy
import dataclasses
import datetime
import enum
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

IN_CLUSTER_DNS_NAME = "ml-pipeline.{}.svc.cluster.local:8888"
KUBE_PROXY_PATH = "api/v1/namespaces/{}/services/ml-pipeline:http/proxy/"
KF_PIPELINES_SA_TOKEN_ENV = "KF_PIPELINES_SA_TOKEN_PATH"
KF_PIPELINES_SA_TOKEN_PATH = "/var/run/secrets/kubeflow/pipelines/token"
ROOT_PARAMETER_NAME = "pipeline-root"

INVALID_CHARACTERS_REGEX = re.compile(r"[^-0-9a-z]+")
MULTIPLE_DASHES_REGEX = re.compile(r"-+")


@dataclasses.dataclass
class Status:
    code: int = 0
    message: str = ""
    details: list[Any] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Duration:
    seconds: int = 0
    nanos: int = 0


@dataclasses.dataclass
class ValueOrRuntimeParameter:
    constant_value: Optional[dict[str, Any]] = None
    runtime_parameter: Optional[str] = None
    constant: Optional[dict[str, Any]] = None


class PrimitiveTypeEnum(enum.IntEnum):
    PRIMITIVE_TYPE_UNSPECIFIED = 0
    INT = 1
    DOUBLE = 2
    STRING = 3


class ParameterTypeEnum(enum.IntEnum):
    PARAMETER_TYPE_ENUM_UNSPECIFIED = 0
    NUMBER_DOUBLE = 1
    NUMBER_INTEGER = 2
    STRING = 3
    BOOLEAN = 4
    LIST = 5
    STRUCT = 6
    TASK_FINAL_STATUS = 7


@dataclasses.dataclass
class Value:
    int_value: Optional[int] = None
    double_value: Optional[float] = None
    string_value: Optional[str] = None


@dataclasses.dataclass
class PipelineInfo:
    name: str = ""
    display_name: str = ""
    description: str = ""


@dataclasses.dataclass
class RuntimeParameter:
    type: PrimitiveTypeEnum = PrimitiveTypeEnum.PRIMITIVE_TYPE_UNSPECIFIED
    default_value: Optional[Value] = None


@dataclasses.dataclass
class PipelineSpec:
    pipeline_info: PipelineInfo = PipelineInfo()
    deployment_spec: Optional[dict[str, Any]] = None
    sdk_version: str = ""
    schema_version: str = ""
    components: dict[str, "ComponentSpec"] = dataclasses.field(default_factory=dict)
    root: Optional["ComponentSpec"] = None
    default_pipeline_root: str = ""


@dataclasses.dataclass
class ArtifactSpec:
    artifact_type: Optional["ArtifactTypeSchema"] = None
    is_artifact_list: bool = False
    is_optional: bool = False
    description: str = ""


@dataclasses.dataclass
class ParameterSpec:
    # The first 'type' field is deprecated.
    type: PrimitiveTypeEnum = PrimitiveTypeEnum.PRIMITIVE_TYPE_UNSPECIFIED
    parameter_type: ParameterTypeEnum = (
        ParameterTypeEnum.PARAMETER_TYPE_ENUM_UNSPECIFIED
    )
    default_value: Optional[Value] = None
    is_optional: bool = False
    description: str = ""


@dataclasses.dataclass
class ComponentInputsSpec:
    artifacts: dict[str, ArtifactSpec] = dataclasses.field(default_factory=dict)
    parameters: dict[str, ParameterSpec] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ComponentOutputsSpec:
    @dataclasses.dataclass
    class ArtifactSpec:
        artifact_type: Optional["ArtifactTypeSchema"] = None
        # Deprecated fields.
        properties: dict[str, "ValueOrRuntimeParameter"] = dataclasses.field(
            default_factory=dict
        )
        custom_properties: dict[str, "ValueOrRuntimeParameter"] = dataclasses.field(
            default_factory=dict
        )
        metadata: Optional[dict[str, Any]] = None
        is_artifact_list: bool = False
        description: str = ""

    @dataclasses.dataclass
    class ParameterSpec:
        type: PrimitiveTypeEnum = (
            PrimitiveTypeEnum.PRIMITIVE_TYPE_UNSPECIFIED
        )  # deprecated
        parameter_type: ParameterTypeEnum = (
            ParameterTypeEnum.PARAMETER_TYPE_ENUM_UNSPECIFIED
        )
        description: str = ""

    artifacts: dict[str, ArtifactSpec] = dataclasses.field(default_factory=dict)
    parameters: dict[str, ParameterSpec] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ComponentSpec:
    input_definitions: ComponentInputsSpec = ComponentInputsSpec()
    output_definitions: ComponentOutputsSpec = ComponentOutputsSpec()
    # oneof implementation: either dag or executor_label is set.
    dag: Optional["DagSpec"] = None
    executor_label: Optional[str] = None
    single_platform_specs: list["SinglePlatformSpec"] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass
class ArtifactSelectorSpec:
    producer_subtask: str = ""
    output_artifact_key: str = ""


@dataclasses.dataclass
class DagOutputArtifactSpec:
    artifact_selectors: list[ArtifactSelectorSpec] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass
class ParameterSelectorSpec:
    producer_subtask: str = ""
    output_parameter_key: str = ""


@dataclasses.dataclass
class ParameterSelectorsSpec:
    parameter_selectors: list[ParameterSelectorSpec] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass
class MapParameterSelectorsSpec:
    mapped_parameters: dict[str, ParameterSelectorSpec] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class DagOutputParameterSpec:
    # oneof: value_from_parameter or value_from_oneof.
    value_from_parameter: Optional[ParameterSelectorSpec] = None
    value_from_oneof: Optional[ParameterSelectorsSpec] = None


@dataclasses.dataclass
class DagSpec:
    tasks: dict[str, "PipelineTaskSpec"] = dataclasses.field(default_factory=dict)
    outputs: Optional["DagOutputsSpec"] = None


@dataclasses.dataclass
class DagOutputsSpec:
    artifacts: dict[str, DagOutputArtifactSpec] = dataclasses.field(
        default_factory=dict
    )
    parameters: dict[str, DagOutputParameterSpec] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class TaskOutputArtifactSpec:
    producer_task: str = ""
    output_artifact_key: str = ""


class InputArtifactSpec:
    task_output_artifact: Optional[TaskOutputArtifactSpec] = None
    component_input_artifact: Optional[str] = None


@dataclasses.dataclass
class TaskOutputParameterSpec:
    producer_task: str = ""
    output_parameter_key: str = ""


@dataclasses.dataclass
class TaskFinalStatus:
    producer_task: str = ""


@dataclasses.dataclass
class InputParameterSpec:
    # oneof: task_output_parameter, runtime_value, component_input_parameter, or task_final_status.
    task_output_parameter: Optional[TaskOutputParameterSpec] = None
    runtime_value: Optional[ValueOrRuntimeParameter] = None
    component_input_parameter: Optional[str] = None
    task_final_status: Optional[TaskFinalStatus] = None
    parameter_expression_selector: Optional[str] = None


@dataclasses.dataclass
class TaskInputsSpec:
    parameters: dict[str, InputParameterSpec] = dataclasses.field(default_factory=dict)
    artifacts: dict[str, InputArtifactSpec] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class OutputArtifactSpec:
    artifact_type: Optional["ArtifactTypeSchema"] = None
    properties: dict[str, "ValueOrRuntimeParameter"] = dataclasses.field(
        default_factory=dict
    )
    custom_properties: dict[str, "ValueOrRuntimeParameter"] = dataclasses.field(
        default_factory=dict
    )
    metadata: Optional[dict[str, Any]] = None


@dataclasses.dataclass
class OutputParameterSpec:
    type: PrimitiveTypeEnum = PrimitiveTypeEnum.PRIMITIVE_TYPE_UNSPECIFIED


@dataclasses.dataclass
class TaskOutputsSpec:
    parameters: dict[str, OutputParameterSpec] = dataclasses.field(default_factory=dict)
    artifacts: dict[str, OutputArtifactSpec] = dataclasses.field(default_factory=dict)


class PrimitiveType(enum.IntEnum):
    PRIMITIVE_TYPE_UNSPECIFIED = 0
    INT = 1
    DOUBLE = 2
    STRING = 3


class ParameterType(enum.IntEnum):
    PARAMETER_TYPE_ENUM_UNSPECIFIED = 0
    NUMBER_DOUBLE = 1
    NUMBER_INTEGER = 2
    STRING = 3
    BOOLEAN = 4
    LIST = 5
    STRUCT = 6
    TASK_FINAL_STATUS = 7


@dataclasses.dataclass
class PipelineTaskInfo:
    name: str = ""


@dataclasses.dataclass
class ComponentRef:
    name: str = ""


@dataclasses.dataclass
class ValueOrRuntimeParameter:
    constant_value: Optional[Value] = None
    runtime_parameter: Optional[str] = None
    constant: Optional[Value] = None


class TriggerStrategy(enum.IntEnum):
    TRIGGER_STRATEGY_UNSPECIFIED = 0
    ALL_UPSTREAM_TASKS_SUCCEEDED = 1
    ALL_UPSTREAM_TASKS_COMPLETED = 2


@dataclasses.dataclass
class PipelineTaskSpec:
    task_info: PipelineTaskInfo = PipelineTaskInfo()
    inputs: TaskInputsSpec = TaskInputsSpec()
    dependent_tasks: list[str] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class CachingOptions:
        enable_cache: bool = False
        cache_key: str = ""

    caching_options: Optional["PipelineTaskSpec.CachingOptions"] = None
    component_ref: Optional[ComponentRef] = None

    @dataclasses.dataclass
    class TriggerPolicy:
        condition: str = ""

        strategy: TriggerStrategy = TriggerStrategy.TRIGGER_STRATEGY_UNSPECIFIED

    trigger_policy: Optional["PipelineTaskSpec.TriggerPolicy"] = None
    # oneof iterator: either artifact_iterator or parameter_iterator.
    artifact_iterator: Optional["ArtifactIteratorSpec"] = None
    parameter_iterator: Optional["ParameterIteratorSpec"] = None

    @dataclasses.dataclass
    class RetryPolicy:
        max_retry_count: int = 0
        backoff_duration: Optional[Duration] = None
        backoff_factor: float = 2.0
        backoff_max_duration: Optional[Duration] = None

    retry_policy: Optional["PipelineTaskSpec.RetryPolicy"] = None

    @dataclasses.dataclass
    class IteratorPolicy:
        parallelism_limit: int = 0

    iterator_policy: Optional["PipelineTaskSpec.IteratorPolicy"] = None


# ------------------------------------------------------------------------------
# ArtifactIteratorSpec and ParameterIteratorSpec
# ------------------------------------------------------------------------------
@dataclasses.dataclass
class ArtifactIteratorSpec:
    @dataclasses.dataclass
    class ItemsSpec:
        input_artifact: str = ""

    items: "ArtifactIteratorSpec.ItemsSpec" = dataclasses.field(
        default_factory=ItemsSpec
    )
    item_input: str = ""


@dataclasses.dataclass
class ParameterIteratorSpec:
    @dataclasses.dataclass
    class ItemsSpec:
        # oneof: raw or input_parameter.
        raw: Optional[str] = None
        input_parameter: Optional[str] = None

    items: "ParameterIteratorSpec.ItemsSpec" = dataclasses.field(
        default_factory=ItemsSpec
    )
    item_input: str = ""


@dataclasses.dataclass
class ArtifactTypeSchema:
    schema_title: Optional[str] = None
    schema_uri: Optional[str] = None  # deprecated
    instance_schema: Optional[str] = None
    schema_version: str = ""


@dataclasses.dataclass
class EnvVar:
    name: str = ""
    value: str = ""


@dataclasses.dataclass
class Exec:
    command: list[str] = dataclasses.field(default_factory=list)
    args: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Lifecycle:
    pre_cache_check: Optional[Exec] = None


@dataclasses.dataclass
class PipelineContainerSpec:
    image: str = ""
    command: list[str] = dataclasses.field(default_factory=list)
    args: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AcceleratorConfig:
    type: Optional[str] = None  # deprecated
    count: Optional[int] = None  # deprecated
    resource_type: str = ""
    resource_count: str = ""


@dataclasses.dataclass
class ResourceSpec:
    cpu_limit: Optional[float] = None  # deprecated
    memory_limit: Optional[float] = None  # deprecated
    cpu_request: Optional[float] = None  # deprecated
    memory_request: Optional[float] = None  # deprecated
    resource_cpu_limit: str = ""
    resource_memory_limit: str = ""
    resource_cpu_request: str = ""
    resource_memory_request: str = ""

    accelerator: Optional[AcceleratorConfig] = None


@dataclasses.dataclass
class ImporterSpec:
    artifact_uri: Optional[ValueOrRuntimeParameter] = None
    type_schema: Optional[ArtifactTypeSchema] = None
    properties: dict[str, ValueOrRuntimeParameter] = dataclasses.field(
        default_factory=dict
    )  # deprecated
    custom_properties: dict[str, ValueOrRuntimeParameter] = dataclasses.field(
        default_factory=dict
    )  # deprecated
    metadata: Optional[dict[str, Any]] = None
    reimport: bool = False


@dataclasses.dataclass
class PipelineDeploymentConfig:
    lifecycle: Optional[Lifecycle] = None
    resources: Optional[ResourceSpec] = None
    env: list[EnvVar] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ArtifactQuerySpec:
    filter: str = ""
    limit: int = 1


@dataclasses.dataclass
class ResolverSpec:
    output_artifact_queries: dict[str, ArtifactQuerySpec] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class AIPlatformCustomJobSpec:
    custom_job: Optional[dict[str, Any]] = None

    executors: dict[str, "ExecutorSpec"] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ExecutorSpec:
    container: Optional[PipelineContainerSpec] = None
    importer: Optional[ImporterSpec] = None
    resolver: Optional[ResolverSpec] = None
    custom_job: Optional[AIPlatformCustomJobSpec] = None


# ------------------------------------------------------------------------------
# RuntimeArtifact and Artifactlist
# ------------------------------------------------------------------------------
@dataclasses.dataclass
class RuntimeArtifact:
    name: str = ""
    type: Optional[ArtifactTypeSchema] = None
    uri: str = ""
    properties: dict[str, Value] = dataclasses.field(default_factory=dict)  # deprecated
    custom_properties: dict[str, Value] = dataclasses.field(
        default_factory=dict
    )  # deprecated
    metadata: Optional[dict[str, Any]] = None


@dataclasses.dataclass
class Artifactlist:
    artifacts: list[RuntimeArtifact] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Inputs:
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)  # deprecated
    artifacts: dict[str, Artifactlist] = dataclasses.field(default_factory=dict)
    parameter_values: dict[str, Value] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class OutputParameter:
    output_file: str = ""


@dataclasses.dataclass
class Outputs:
    parameters: dict[str, OutputParameter] = dataclasses.field(default_factory=dict)
    artifacts: dict[str, Artifactlist] = dataclasses.field(default_factory=dict)
    output_file: str = ""


@dataclasses.dataclass
class ExecutorInput:
    inputs: Optional[Inputs] = None
    outputs: Optional[Outputs] = None


@dataclasses.dataclass
class ExecutorOutput:
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)  # deprecated
    artifacts: dict[str, Artifactlist] = dataclasses.field(default_factory=dict)
    parameter_values: dict[str, Value] = dataclasses.field(default_factory=dict)


# ------------------------------------------------------------------------------
# PipelineTaskFinalStatus and PipelineStateEnum
# ------------------------------------------------------------------------------
@dataclasses.dataclass
class PipelineTaskFinalStatus:
    state: str = ""
    error: Optional[Status] = None
    pipeline_job_uuid: Optional[int] = None  # deprecated
    pipeline_job_name: str = ""
    pipeline_job_resource_name: str = ""
    pipeline_task_name: str = ""


class PipelineTaskState(enum.IntEnum):
    TASK_STATE_UNSPECIFIED = 0
    PENDING = 1
    RUNNING_DRIVER = 2
    DRIVER_SUCCEEDED = 3
    RUNNING_EXECUTOR = 4
    SUCCEEDED = 5
    CANCEL_PENDING = 6
    CANCELLING = 7
    CANCELLED = 8
    FAILED = 9
    SKIPPED = 10
    QUEUED = 11
    NOT_TRIGGERED = 12
    UNSCHEDULABLE = 13


# ------------------------------------------------------------------------------
# PlatformSpec, SinglePlatformSpec, PlatformDeploymentConfig, and PipelineConfig
# ------------------------------------------------------------------------------
@dataclasses.dataclass
class PlatformSpec:
    platforms: dict[str, "SinglePlatformSpec"] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PipelineDoc:
    pipeline_spec: PipelineSpec
    platform_spec: PlatformSpec

    def to_dict(self) -> dict:
        return {
            "pipeline_spec": orjson.dumps(self.pipeline_spec),
            "platform_spec": orjson.dumps(self.platform_spec),
        }


@dataclasses.dataclass
class SinglePlatformSpec:
    deployment_spec: Optional["PlatformDeploymentConfig"] = None
    platform: str = ""
    config: Optional[dict[str, Any]] = None
    pipelineConfig: Optional["PipelineConfig"] = None


@dataclasses.dataclass
class PlatformDeploymentConfig:
    executors: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PipelineConfig:
    semaphore_key: str = ""
    mutex_name: str = ""


@dataclasses.dataclass
class _JobConfig:
    pipeline_spec: dict
    pipeline_version_reference: kfp_server_api.V2beta1PipelineVersionReference
    runtime_config: kfp_server_api.V2beta1RuntimeConfig


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

        # Fallback: try local kubeconfig
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
        # Pass the namespace as a separate parameter if your API supports it.
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
                (see `filter.proto message <https://github.com/kubeflow/pipelines/blob/cb7d9a87c999eb1d2280959e5afbeee9e270ef3d/backend/api/v2beta1/filter.proto>`_). For a list of all filter operations ``'opertion'``, see `here <https://github.com/kubeflow/pipelines/blob/777c98153daf3dfae82730e14ff37bdddc334c4d/sdk/python/kfp/client/client.py#L37-L45>`_. Example:

                  ::

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

        pipeline_spec = existing_run.pipeline_spec
        if not (pipeline_spec.pipeline_id or pipeline_spec.workflow_manifest):
            raise ValueError(
                "The original run does not have a pipeline_id or workflow_manifest."
            )

        # Save the old manifest to a temporary file if there's no pipeline_id
        workflow_manifest_path = None
        if not pipeline_spec.pipeline_id:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as temp_file:
                temp_file.write(pipeline_spec.workflow_manifest)
                workflow_manifest_path = temp_file.name

        # Convert parameters from list => dict if needed:
        parameters = {}
        if pipeline_spec.parameters:
            # Usually parameters might be a list of ApiParameter, e.g. [ApiParameter(name="p", value="v"), ...]
            if isinstance(pipeline_spec.parameters, list):
                for p in pipeline_spec.parameters:
                    parameters[p.name] = p.value
            else:
                parameters = pipeline_spec.parameters

        # Generate a new run name
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

    def _create_job_config(
        self,
        params: Optional[dict[str, Any]],
        pipeline_package_path: Optional[str],
        pipeline_id: Optional[str],
        version_id: Optional[str],
        enable_caching: Optional[bool],
        cache_key: Optional[str],
        pipeline_root: Optional[str],
    ) -> _JobConfig:
        """Creates a JobConfig with spec and resource_references.

        Args:
            pipeline_package_path: Local path of the pipeline package (the
                filename should end with one of the following .tar.gz, .tgz,
                .zip, .yaml, .yml).
            params: A dictionary with key as param name and value as param value.
            pipeline_id: ID of a pipeline.
            version_id: ID of a pipeline version.
                If both pipeline_id and version_id are specified, version_id
                will take precedence. If only pipeline_id is specified, the
                default version of this pipeline is used to create the run.
            enable_caching: Whether or not to enable caching for the
                run. If not set, defaults to the compile time settings, which
                is ``True`` for all tasks by default, while users may specify
                different caching options for individual tasks. If set, the
                setting applies to all tasks in the pipeline (overrides the
                compile time settings).
            cache_key (optional): Customized cache key for this task.
                If set, the cache_key will be used as the key for the task's cache.
            pipeline_root: Root path of the pipeline outputs.

        Returns:
            A _JobConfig object with attributes .pipeline_spec,
                .pipeline_version_reference, and .runtime_config.
        """
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

            # Caching option set at submission time overrides the compile time
            # settings.
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
        return _JobConfig(
            pipeline_spec=pipeline_spec,
            pipeline_version_reference=pipeline_version_reference,
            runtime_config=runtime_config,
        )

    @staticmethod
    def _parse_pipeline_obj(package_file: str) -> Any:
        def _choose_pipeline_yaml_file(file_list: list[str]) -> str:
            pipeline_file = "pipeline.yaml"
            yaml_files = [f for f in file_list if f.endswith(".yaml")]
            if not yaml_files:
                raise ValueError("No .yaml file found in package.")
            if pipeline_file in yaml_files:
                return pipeline_file
            if len(yaml_files) == 1:
                return yaml_files[0]
            raise ValueError("Multiple .yaml files found and none named pipeline.yaml.")

        if package_file.endswith((".tar.gz", ".tgz")):
            with tarfile.open(package_file, "r:gz") as tar:
                file_names = [m.name for m in tar if m.isfile()]
                chosen_file = _choose_pipeline_yaml_file(file_names)
                with tar.extractfile(chosen_file) as f:
                    return yaml.safe_load(f)
        elif package_file.endswith(".zip"):
            with zipfile.ZipFile(package_file, "r") as zf:
                chosen_file = _choose_pipeline_yaml_file(zf.namelist())
                with zf.open(chosen_file) as f:
                    return yaml.safe_load(f)
        elif package_file.endswith((".yaml", ".yml")):
            with open(package_file) as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("package_file must be .tar.gz, .tgz, .zip, .yaml, or .yml")


def _extract_pipeline_yaml(package_file: str) -> PipelineDoc:
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

    def _safe_load_yaml(stream: TextIO) -> PipelineDoc:
        docs = yaml.safe_load_all(stream)
        pipeline_spec_dict = None
        platform_spec_dict = {}
        for doc in docs:
            if pipeline_spec_dict is None:
                pipeline_spec_dict = doc
            else:
                platform_spec_dict.update(doc)

        return PipelineDoc(
            pipeline_spec=PipelineSpec(**pipeline_spec_dict),
            platform_spec=PlatformSpec(**platform_spec_dict),
        )

    if package_file.endswith(".tar.gz") or package_file.endswith(".tgz"):
        with tarfile.open(package_file, "r:gz") as tar:
            file_names = [member.name for member in tar if member.isfile()]
            pipeline_file = _choose_pipeline_file(file_names)
            with tar.extractfile(tar.getmember(pipeline_file)) as f:  # type: ignore
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
    pipeline_spec: PipelineSpec,
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
