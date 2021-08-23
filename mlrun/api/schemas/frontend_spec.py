import enum
import typing

import pydantic


class ProjectMembershipFeatureFlag(str, enum.Enum):
    enabled = "enabled"
    disabled = "disabled"


class FeatureFlags(pydantic.BaseModel):
    project_membership: ProjectMembershipFeatureFlag


class FrontendSpec(pydantic.BaseModel):
    jobs_dashboard_url: typing.Optional[str]
    abortable_function_kinds: typing.List[str] = []
    feature_flags: FeatureFlags
    default_function_priority_class_name: typing.Optional[str]
    valid_function_priority_class_names: typing.List[str] = []
