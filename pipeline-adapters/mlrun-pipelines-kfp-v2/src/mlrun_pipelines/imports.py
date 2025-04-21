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

import logging
from types import ModuleType
from typing import Any, Optional

from mlrun_pipelines.common.imports import (
    dsl,
    kfp,
)

logger = logging.getLogger(__name__)

# Initialize placeholders for KFP v2 components
kubernetes: ModuleType = ModuleType("kubernetes")
PipelineTask: Optional[type["PipelineTaskBase"]] = None


class PipelineTaskBase:
    name: str
    command: list
    args: list[Any]
    file_outputs: dict[str, str]

    def add_env_variable(self, env_var: Any) -> "PipelineTaskBase":
        self.container.env.append(env_var)
        return self


class DummyPipelineTask(PipelineTaskBase):
    def __init__(
        self,
        name: str,
        command: list,
        args: Optional[list[Any]] = None,
        file_outputs: Optional[dict[str, str]] = None,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.file_outputs = file_outputs or {}

    def add_env_variable(self, env_var: Any):
        self.container.env.append(env_var)
        return self


# Try importing the actual KFP v2 components
try:
    import kfp as real_kfp
    import kfp.compiler as real_compiler
    import kfp.dsl as real_dsl
    from kfp import Client as real_Client
    from kfp.dsl import PipelineTask as real_PipelineTask

    # Assign real KFP components
    kfp = real_kfp
    dsl = real_dsl
    compiler = real_compiler
    Compiler = real_compiler.Compiler
    PipelineTask = real_PipelineTask
    Client = real_Client
    kfp.Client = Client

except ImportError:
    logger.warning(
        "Kubeflow Pipelines (KFP) is not installed. Using noop implementations."
    )
    from mlrun_pipelines.common.imports import (
        Compiler,
        compiler,
        dsl,
        kfp,
    )

    PipelineTask = DummyPipelineTask
    dsl.PipelineTask = DummyPipelineTask

__all__ = [
    "Client",
    "Compiler",
    "PipelineTask",
    "compiler",
    "dsl",
    "kfp",
]
