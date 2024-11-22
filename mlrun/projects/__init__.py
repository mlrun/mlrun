# Copyright 2023 Iguazio
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

# Don't remove this, used by sphinx documentation
__all__ = [
    "load_project",
    "new_project",
    "get_or_create_project",
    "MlrunProject",
    "ProjectMetadata",
    "ProjectSpec",
    "ProjectStatus",
    "run_function",
    "build_function",
    "deploy_function",
]

from .operations import build_function, deploy_function, run_function  # noqa
from .pipelines import (
    import_remote_project,
    load_and_run_workflow,
    load_and_run,
    pipeline_context,
)  # noqa
from .project import (
    MlrunProject,
    ProjectMetadata,
    ProjectSpec,
    ProjectStatus,
    get_or_create_project,
    load_project,
    new_project,
)
