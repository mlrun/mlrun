# Copyright 2024 Iguazio
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
from __future__ import annotations

import dataclasses
import enum
from collections.abc import Generator
from typing import Any, Optional

import orjson
from kfp_server_api import V2beta1PipelineVersionReference, V2beta1RuntimeConfig

from mlrun_pipelines.common.helpers import FlexibleMapper
from mlrun_pipelines.imports import PipelineTask

# class pointer for type checking on the main MLRun codebase
PipelineNodeWrapper = PipelineTask


class PipelineStep(FlexibleMapper):
    @property
    def step_type(self):
        raise NotImplementedError

    @property
    def node_name(self):
        raise NotImplementedError

    @property
    def phase(self):
        raise NotImplementedError

    @property
    def skipped(self):
        raise NotImplementedError

    @property
    def display_name(self):
        raise NotImplementedError

    def get_annotation(self, annotation_name: str):
        raise NotImplementedError


class PipelineManifest(FlexibleMapper):
    """
    A Pipeline Manifest might have been created by an 1.8 SDK regardless of coming from a 2.0 API,
    so this class tries to account for that
    """

    def get_schema_version(self) -> str:
        try:
            return self._external_data["schemaVersion"]
        except KeyError:
            return self._external_data["apiVersion"]

    def is_argo_compatible(self) -> bool:
        # TODO: make sure this is compatible with KFP 2. The schema version in kfp 2 is a semver string,
        #       but since this code supports kfp 1.8 as well, where it considers the api version as the schema version
        #       we need to check if the schema version starts with "argoproj.io". Either way, for now this check is
        #       good enough and won't break whether the schema version is a semver string or not.
        schema_version_split = self.get_schema_version().split("/")[0]
        return schema_version_split == "argoproj.io"

    def get_executors(self):
        if self.is_argo_compatible():
            yield from [
                (t.get("name"), t) for t in self._external_data["spec"]["templates"]
            ]
        else:
            yield from self._external_data["deploymentSpec"]["executors"].items()

    def get_steps(self) -> Generator[PipelineStep, None, None]:
        raise NotImplementedError


class PipelineRun(FlexibleMapper):
    @property
    def id(self):
        return self._external_data["run_id"]

    @property
    def name(self):
        return self._external_data["display_name"]

    @name.setter
    def name(self, name):
        self._external_data["display_name"] = name

    @property
    def status(self):
        return self._external_data["state"]

    @status.setter
    def status(self, status):
        self._external_data["state"] = status

    @property
    def description(self):
        return self._external_data["description"]

    @description.setter
    def description(self, description):
        self._external_data["description"] = description

    @property
    def created_at(self):
        return self._external_data["created_at"]

    @created_at.setter
    def created_at(self, created_at):
        self._external_data["created_at"] = created_at

    @property
    def scheduled_at(self):
        return self._external_data["scheduled_at"]

    @scheduled_at.setter
    def scheduled_at(self, scheduled_at):
        self._external_data["scheduled_at"] = scheduled_at

    @property
    def finished_at(self):
        return self._external_data["finished_at"]

    @finished_at.setter
    def finished_at(self, finished_at):
        self._external_data["finished_at"] = finished_at

    def experiment_id(self) -> str:
        raise NotImplementedError

    def workflow_manifest(self) -> PipelineManifest:
        return PipelineManifest(
            self._external_data["pipeline_spec"],
        )


class PipelineExperiment(FlexibleMapper):
    @property
    def id(self):
        return self._external_data["experiment_id"]


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


@dataclasses.dataclass
class Value:
    int_value: Optional[int] = None
    double_value: Optional[float] = None
    string_value: Optional[str] = None


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
class Value:
    int_value: Optional[int] = None
    double_value: Optional[float] = None
    string_value: Optional[str] = None


@dataclasses.dataclass
class ValueOrRuntimeParameter:
    constant_value: Optional[Value] = None
    runtime_parameter: Optional[str] = None
    constant: Optional[Value] = None

    def __post_init__(self):
        fields_set = sum(
            x is not None
            for x in (self.constant_value, self.runtime_parameter, self.constant)
        )
        if fields_set > 1:
            raise ValueError(
                "Only one of constant_value, runtime_parameter, or constant can be set"
            )


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
    components: dict[str, ComponentSpec] = dataclasses.field(default_factory=dict)
    root: Optional[ComponentSpec] = None
    default_pipeline_root: str = ""


@dataclasses.dataclass
class ArtifactSpec:
    artifact_type: Optional[ArtifactTypeSchema] = None
    is_artifact_list: bool = False
    is_optional: bool = False
    description: str = ""


@dataclasses.dataclass
class ParameterSpec:
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
        artifact_type: Optional[ArtifactTypeSchema] = None
        properties: dict[str, ValueOrRuntimeParameter] = dataclasses.field(
            default_factory=dict
        )
        custom_properties: dict[str, ValueOrRuntimeParameter] = dataclasses.field(
            default_factory=dict
        )
        metadata: Optional[dict[str, object]] = None
        is_artifact_list: bool = False
        description: str = ""

    @dataclasses.dataclass
    class ParameterSpec:
        parameter_type: ParameterTypeEnum = (
            ParameterTypeEnum.PARAMETER_TYPE_ENUM_UNSPECIFIED
        )
        description: str = ""

    artifacts: dict[str, ComponentOutputsSpec.ArtifactSpec] = dataclasses.field(
        default_factory=dict
    )
    parameters: dict[str, ComponentOutputsSpec.ParameterSpec] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class ComponentSpec:
    input_definitions: ComponentInputsSpec = ComponentInputsSpec()
    output_definitions: ComponentOutputsSpec = ComponentOutputsSpec()
    dag: Optional[DagSpec] = None
    executor_label: Optional[str] = None
    single_platform_specs: list[SinglePlatformSpec] = dataclasses.field(
        default_factory=list
    )

    def __post_init__(self):
        if self.dag is not None and self.executor_label is not None:
            raise ValueError("Only one of dag or executor_label can be set")


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
    value_from_parameter: Optional[ParameterSelectorSpec] = None
    value_from_oneof: Optional[ParameterSelectorsSpec] = None

    def __post_init__(self):
        if self.value_from_parameter is not None and self.value_from_oneof is not None:
            raise ValueError(
                "Only one of value_from_parameter or value_from_oneof can be set"
            )


@dataclasses.dataclass
class DagSpec:
    tasks: dict[str, PipelineTaskSpec] = dataclasses.field(default_factory=dict)
    outputs: Optional[DagOutputsSpec] = None


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
    task_output_parameter: Optional[TaskOutputParameterSpec] = None
    runtime_value: Optional[ValueOrRuntimeParameter] = None
    component_input_parameter: Optional[str] = None
    task_final_status: Optional[TaskFinalStatus] = None
    parameter_expression_selector: Optional[str] = None

    def __post_init__(self):
        oneof_fields = [
            self.task_output_parameter,
            self.runtime_value,
            self.component_input_parameter,
            self.task_final_status,
        ]
        if sum(field is not None for field in oneof_fields) > 1:
            raise ValueError(
                "Only one of task_output_parameter, runtime_value, component_input_parameter, or task_final_status can be set"
            )


@dataclasses.dataclass
class TaskInputsSpec:
    parameters: dict[str, InputParameterSpec] = dataclasses.field(default_factory=dict)
    artifacts: dict[str, InputArtifactSpec] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class OutputArtifactSpec:
    artifact_type: Optional[ArtifactTypeSchema] = None
    properties: dict[str, ValueOrRuntimeParameter] = dataclasses.field(
        default_factory=dict
    )
    custom_properties: dict[str, ValueOrRuntimeParameter] = dataclasses.field(
        default_factory=dict
    )
    metadata: Optional[dict[str, object]] = None


@dataclasses.dataclass
class OutputParameterSpec:
    type: PrimitiveTypeEnum = PrimitiveTypeEnum.PRIMITIVE_TYPE_UNSPECIFIED


@dataclasses.dataclass
class TaskOutputsSpec:
    parameters: dict[str, OutputParameterSpec] = dataclasses.field(default_factory=dict)
    artifacts: dict[str, OutputArtifactSpec] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PipelineTaskInfo:
    name: str = ""


@dataclasses.dataclass
class ComponentRef:
    name: str = ""


class TriggerStrategy(enum.IntEnum):
    TRIGGER_STRATEGY_UNSPECIFIED = 0
    ALL_UPSTREAM_TASKS_SUCCEEDED = 1
    ALL_UPSTREAM_TASKS_COMPLETED = 2


@dataclasses.dataclass
class PipelineTaskSpec:
    task_info: PipelineTaskInfo = PipelineTaskInfo()
    inputs: TaskInputsSpec = TaskInputsSpec()
    dependent_tasks: list[str] = dataclasses.field(default_factory=list)
    artifact_iterator: Optional[ArtifactIteratorSpec] = None
    parameter_iterator: Optional[ParameterIteratorSpec] = None
    caching_options: Optional[CachingOptions] = None
    component_ref: Optional[ComponentRef] = None
    trigger_policy: Optional[TriggerPolicy] = None
    retry_policy: Optional[RetryPolicy] = None
    iterator_policy: Optional[IteratorPolicy] = None

    def __post_init__(self):
        if self.artifact_iterator is not None and self.parameter_iterator is not None:
            raise ValueError(
                "Only one of artifact_iterator or parameter_iterator can be set"
            )

    @dataclasses.dataclass
    class CachingOptions:
        enable_cache: bool = False
        cache_key: str = ""

    @dataclasses.dataclass
    class TriggerPolicy:
        condition: str = ""
        strategy: TriggerStrategy = TriggerStrategy.TRIGGER_STRATEGY_UNSPECIFIED

    @dataclasses.dataclass
    class RetryPolicy:
        max_retry_count: int = 0
        backoff_duration: Optional[Duration] = None
        backoff_factor: float = 2.0
        backoff_max_duration: Optional[Duration] = None

    @dataclasses.dataclass
    class IteratorPolicy:
        parallelism_limit: int = 0


@dataclasses.dataclass
class ItemsSpec:
    input_artifact: str = ""


@dataclasses.dataclass
class ArtifactIteratorSpec:
    items: ItemsSpec = dataclasses.field(default_factory=ItemsSpec)
    item_input: str = ""


@dataclasses.dataclass
class ParameterIteratorSpec:
    @dataclasses.dataclass
    class ItemsSpec:
        raw: Optional[str] = None
        input_parameter: Optional[str] = None

        def __post_init__(self):
            if self.raw is not None and self.input_parameter is not None:
                raise ValueError("Only one of raw or input_parameter can be set")

    items: ParameterIteratorSpec.ItemsSpec = dataclasses.field(
        default_factory=lambda: ParameterIteratorSpec.ItemsSpec()
    )
    item_input: str = ""


@dataclasses.dataclass
class ArtifactTypeSchema:
    schema_title: Optional[str] = None
    schema_uri: Optional[str] = None
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
    type: Optional[str] = None
    count: Optional[int] = None
    resource_type: str = ""
    resource_count: str = ""


@dataclasses.dataclass
class ResourceSpec:
    cpu_limit: Optional[float] = None
    memory_limit: Optional[float] = None
    cpu_request: Optional[float] = None
    memory_request: Optional[float] = None
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
    )
    custom_properties: dict[str, ValueOrRuntimeParameter] = dataclasses.field(
        default_factory=dict
    )
    metadata: Optional[dict[str, object]] = None
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
    custom_job: Optional[dict[str, object]] = None
    executors: dict[str, ExecutorSpec] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ExecutorSpec:
    container: Optional[PipelineContainerSpec] = None
    importer: Optional[ImporterSpec] = None
    resolver: Optional[ResolverSpec] = None
    custom_job: Optional[AIPlatformCustomJobSpec] = None


@dataclasses.dataclass
class RuntimeArtifact:
    name: str = ""
    type: Optional[ArtifactTypeSchema] = None
    uri: str = ""
    properties: dict[str, Value] = dataclasses.field(default_factory=dict)
    custom_properties: dict[str, Value] = dataclasses.field(default_factory=dict)
    metadata: Optional[dict[str, object]] = None


@dataclasses.dataclass
class Artifactlist:
    artifacts: list[RuntimeArtifact] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Inputs:
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)
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
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)
    artifacts: dict[str, Artifactlist] = dataclasses.field(default_factory=dict)
    parameter_values: dict[str, Value] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PipelineTaskFinalStatus:
    state: str = ""
    error: Optional[Status] = None
    pipeline_job_uuid: Optional[int] = None
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


@dataclasses.dataclass
class PlatformSpec:
    platforms: dict[str, SinglePlatformSpec] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PipelineDoc:
    pipeline_spec: PipelineSpec
    platform_spec: PlatformSpec

    def to_dict(self) -> dict:
        return {
            "pipeline_spec": orjson.dumps(
                dataclasses.asdict(self.pipeline_spec)
            ).decode("utf-8"),
            "platform_spec": orjson.dumps(
                dataclasses.asdict(self.platform_spec)
            ).decode("utf-8"),
        }


@dataclasses.dataclass
class SinglePlatformSpec:
    deployment_spec: Optional[PlatformDeploymentConfig] = None
    platform: str = ""
    config: Optional[dict[str, object]] = None
    pipelineConfig: Optional[PipelineConfig] = None


@dataclasses.dataclass
class PlatformDeploymentConfig:
    executors: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PipelineConfig:
    semaphore_key: str = ""
    mutex_name: str = ""


@dataclasses.dataclass
class JobConfig:
    pipeline_spec: dict
    pipeline_version_reference: V2beta1PipelineVersionReference
    runtime_config: V2beta1RuntimeConfig
