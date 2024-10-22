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
import logging
from types import ModuleType
from typing import Any, Optional

from mlrun_pipelines.common.imports import (
    DummyClient,
    DummyCompiler,
    DummyContainerOp,
    DummyDSL,
    DummyPipelineConf,
    DummyPipelineParam,
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

    def add_env_variable(self, name: str, value: str) -> None:
        pass


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
        logger.warning(
            f"[NoOp] PipelineTask created with name='{name}', command={command}, args={self.args}, "
            f"file_outputs={self.file_outputs}"
        )

    def add_env_variable(self, name: str, value: str) -> None:
        logger.warning(
            f"[NoOp] add_env_variable called with name='{name}', value='{value}'"
        )
        pass


class DummyKubernetes:
    def __init__(self) -> None:
        logger.warning("[NoOp] Kubernetes client initialized but does nothing.")

    def add_pod_annotation(self, name: str, value: str):
        logger.warning(
            f"[NoOp] add_pod_annotation called with name='{name}', value='{value}'"
        )


# Try importing the actual KFP v2 components
try:
    import kfp as real_kfp
    import kfp.compiler as real_compiler
    import kfp.dsl as real_dsl
    import kfp.kubernetes as real_kubernetes
    from kfp import Client as real_Client
    from kfp.dsl import ContainerOp as real_ContainerOp
    from kfp.dsl import PipelineConf as real_PipelineConf
    from kfp.dsl import PipelineParam as real_PipelineParam
    from kfp.dsl import PipelineTask as real_PipelineTask

    # Assign real KFP components
    kfp = real_kfp
    dsl = real_dsl
    compiler = real_compiler
    kubernetes = real_kubernetes
    PipelineTask = real_PipelineTask
    ContainerOp = real_ContainerOp
    Client = real_Client
    PipelineParam = real_PipelineParam
    PipelineConf = real_PipelineConf

except ImportError:
    logger.warning(
        "Kubeflow Pipelines (KFP) v2 is not installed. Using no-operation (noop) implementations."
    )
    PipelineTask = DummyPipelineTask
    PipelineConf = DummyPipelineConf
    PipelineParam = DummyPipelineParam
    ContainerOp = DummyContainerOp
    Client = DummyClient
    dsl = DummyDSL()
    Compiler = DummyCompiler()
    kubernetes = DummyKubernetes()

__all__ = [
    "kfp",
    "dsl",
    "compiler",
    "kubernetes",
    "PipelineTask",
    "ContainerOp",
    "Client",
    "PipelineParam",
    "PipelineConf",
]
