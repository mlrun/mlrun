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
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DummyContainer:
    env: list[dict[str, str]] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    image: str = ""
    resources: dict[str, Any] = field(default_factory=dict)

    def set_command(self, command: list[str]) -> None:
        logger.debug(f"[NoOp] set_command called with command={command}")
        self.command = command

    def set_args(self, args: list[str]) -> None:
        logger.debug(f"[NoOp] set_args called with args={args}")
        self.args = args

    def set_image(self, image: str) -> None:
        logger.debug(f"[NoOp] set_image called with image={image}")
        self.image = image

    def add_env_variable(self, name: str, value: str) -> None:
        logger.debug(
            f"[NoOp] add_env_variable called with name='{name}', value='{value}'"
        )
        self.env.append({"name": name, "value": value})

    def set_resources(self, resources: dict[str, Any]) -> None:
        logger.debug(f"[NoOp] set_resources called with resources={resources}")
        self.resources = resources


@dataclass
class DummyContainerOp:
    name: str
    image: str
    command: list[str]
    file_outputs: Optional[dict[str, str]] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    pod_labels: dict[str, str] = field(default_factory=dict)
    pod_annotations: dict[str, str] = field(default_factory=dict)
    volumes: list[dict[str, Any]] = field(default_factory=list)
    container: DummyContainer = field(default_factory=DummyContainer)

    def add_pod_label(self, key: str, value: str) -> None:
        logger.debug(f"[NoOp] add_pod_label called with key='{key}', value='{value}'")
        self.pod_labels[key] = value

    def add_volume(self, *args: Any, **kwargs: Any) -> None:
        logger.debug(f"[NoOp] add_volume called with args={args}, kwargs={kwargs}")
        self.volumes.append({"args": args, "kwargs": kwargs})

    def add_env_variable(self, name: str, value: str) -> None:
        logger.debug(
            f"[NoOp] add_env_variable called with name='{name}', value='{value}'"
        )
        self.container.env.append({"name": name, "value": value})


try:
    import kfp as real_kfp
    import kfp.compiler as real_compiler
    import kfp.dsl as real_dsl
    from kfp import Client as real_Client
    from kfp.dsl import ContainerOp as real_ContainerOp
    from kfp.dsl import PipelineConf as real_PipelineConf
    from kfp.dsl import PipelineParam as real_PipelineParam

    # Assign real KFP components
    kfp = real_kfp
    dsl = real_dsl
    compiler = real_compiler
    Compiler = real_compiler.Compiler
    ContainerOp = real_ContainerOp
    Client = real_Client
    PipelineParam = real_PipelineParam
    PipelineConf = real_PipelineConf

    if hasattr(ContainerOp, "_DISABLE_REUSABLE_COMPONENT_WARNING"):
        ContainerOp._DISABLE_REUSABLE_COMPONENT_WARNING = True

except ImportError:
    logger.warning(
        "Kubeflow Pipelines (KFP) is not installed. Using noop implementations."
    )
    from mlrun_pipelines.common.imports import (
        Client,
        Compiler,
        PipelineConf,
        PipelineParam,
        compiler,
        dsl,
        kfp,
    )

    ContainerOp = DummyContainerOp


__all__ = [
    "Client",
    "Compiler",
    "ContainerOp",
    "PipelineConf",
    "PipelineParam",
    "compiler",
    "dsl",
    "kfp",
]
