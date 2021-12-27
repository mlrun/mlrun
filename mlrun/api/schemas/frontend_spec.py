import enum
import typing

import pydantic


class ProjectMembershipFeatureFlag(str, enum.Enum):
    enabled = "enabled"
    disabled = "disabled"


class AuthenticationFeatureFlag(str, enum.Enum):
    none = "none"
    basic = "basic"
    bearer = "bearer"
    iguazio = "iguazio"


class FeatureFlags(pydantic.BaseModel):
    project_membership: ProjectMembershipFeatureFlag
    authentication: AuthenticationFeatureFlag


class FrontendSpec(pydantic.BaseModel):
    jobs_dashboard_url: typing.Optional[str]
    abortable_function_kinds: typing.List[str] = []
    feature_flags: FeatureFlags
    default_function_priority_class_name: typing.Optional[str]
    valid_function_priority_class_names: typing.List[str] = []
    default_function_image_by_kind: typing.Dict[str, str] = {}
    function_deployment_target_image_template: typing.Optional[str]
    function_deployment_mlrun_command: typing.Optional[str]
    auto_mount_type: typing.Optional[str]
    auto_mount_params: typing.Dict[str, str] = {}
