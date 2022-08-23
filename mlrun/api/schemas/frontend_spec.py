# Copyright 2018 Iguazio
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
