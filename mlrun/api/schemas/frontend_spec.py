import enum
import typing

import pydantic

from .k8s import Resources


class ProjectMembershipFeatureFlag(str, enum.Enum):
    enabled = "enabled"
    disabled = "disabled"


class PreemptionNodesFeatureFlag(str, enum.Enum):
    enabled = "enabled"
    disabled = "disabled"


class AuthenticationFeatureFlag(str, enum.Enum):
    none = "none"
    basic = "basic"
    bearer = "bearer"
    iguazio = "iguazio"


class NuclioStreamsFeatureFlag(str, enum.Enum):
    enabled = "enabled"
    disabled = "disabled"


class FeatureFlags(pydantic.BaseModel):
    project_membership: ProjectMembershipFeatureFlag
    authentication: AuthenticationFeatureFlag
    nuclio_streams: NuclioStreamsFeatureFlag
    preemption_nodes: PreemptionNodesFeatureFlag


class FrontendSpec(pydantic.BaseModel):
    jobs_dashboard_url: typing.Optional[str]
    abortable_function_kinds: typing.List[str] = []
    feature_flags: FeatureFlags
    default_function_priority_class_name: typing.Optional[str]
    valid_function_priority_class_names: typing.List[str] = []
    default_function_image_by_kind: typing.Dict[str, str] = {}
    function_deployment_target_image_template: typing.Optional[str]
    function_deployment_target_image_name_prefix_template: str
    function_deployment_target_image_registries_to_enforce_prefix: typing.List[str] = []
    function_deployment_mlrun_command: typing.Optional[str]
    auto_mount_type: typing.Optional[str]
    auto_mount_params: typing.Dict[str, str] = {}
    default_artifact_path: str
    default_function_pod_resources: Resources = Resources()
    default_function_preemption_mode: str
