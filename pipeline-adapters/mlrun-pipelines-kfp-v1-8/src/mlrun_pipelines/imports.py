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
from dataclasses import dataclass, field
from typing import Any, Optional

from kubernetes.client import V1VolumeMount

logger = logging.getLogger(__name__)


@dataclass
class ContainerResources:
    resources: dict[str, Any] = field(default_factory=dict)
    limit: dict[str, Any] = field(default_factory=dict)


@dataclass
class DummyContainer:
    env: list[dict[str, str]] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    image: str = ""
    resources: ContainerResources = field(default_factory=ContainerResources)
    volume_mounts: list[V1VolumeMount] = field(default_factory=list)

    def add_volume_mount(self, volume_mount: V1VolumeMount) -> "DummyContainer":
        self.volumes.append(volume_mount)
        return self

    def set_command(self, command: list[str]) -> "DummyContainer":
        self.command = command
        return self

    def set_args(self, args: list[str]) -> "DummyContainer":
        self.args = args
        return self

    def set_image(self, image: str) -> "DummyContainer":
        self.image = image
        return self

    def set_env_variable(self, name: str, value: str) -> "DummyContainer":
        self.env.append({"name": name, "value": value})
        return self

    def set_resources(self, resources: dict[str, Any]) -> "DummyContainer":
        self.resources = resources
        return self

    def add_resource_limit(self, resource_name, value) -> "DummyContainer":
        self.resources.limits = self.resources.limits or {}
        self.resources.limits.update({resource_name: value})
        return self

    def add_resource_request(self, resource_name, value) -> "DummyContainer":
        self.resources.requests = self.resources.requests or {}
        self.resources.requests.update({resource_name: value})
        return self


@dataclass
class DummyContainerOp:
    name: str
    image: str
    command: list[str] = field(default_factory=list)
    file_outputs: Optional[dict[str, str]] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    pod_labels: dict[str, str] = field(default_factory=dict)
    pod_annotations: dict[str, str] = field(default_factory=dict)
    node_selector: dict[str, str] = field(default_factory=dict)
    affinity: dict[str, str] = field(default_factory=dict)
    tolerations: dict[str, str] = field(default_factory=dict)
    volumes: list[dict[str, Any]] = field(default_factory=list)
    container: DummyContainer = field(default_factory=DummyContainer)

    def _register_op_handler(self) -> str:
        return ""

    def add_pod_label(self, key: str, value: str):
        self.pod_labels[key] = value
        return self

    def add_pod_annotation(self, key: str, value: str):
        self.pod_annotations[key] = value
        return self

    def add_volume(self, volume: Any):
        self.volumes.append(volume)
        return self

    def add_env_variable(self, env_var: Any):
        self.container.env.append(env_var)
        return self

    def add_node_selector(self, key: str, value: str):
        self.node_selector[key] = value
        return self

    def add_affinity(self, key: str, value: str):
        self.affinity[key] = value
        return self

    def add_toleration(self, key: str, value: str):
        self.tolerations[key] = value
        return self

    def add_file_output(self, key: str, value: str):
        self.file_outputs[key] = value
        return self


try:
    import kfp as real_kfp
    import kfp.compiler as real_compiler
    import kfp.dsl as real_dsl
    from kfp.dsl import ContainerOp as real_ContainerOp
    from kfp.dsl import PipelineConf as real_PipelineConf
    from kfp.dsl import PipelineParam as real_PipelineParam

    # Assign real KFP components
    kfp = real_kfp
    dsl = real_dsl
    compiler = real_compiler
    Compiler = real_compiler.Compiler
    ContainerOp = real_ContainerOp
    PipelineParam = real_PipelineParam
    PipelineConf = real_PipelineConf
    dsl.ContainerOp = ContainerOp

    if hasattr(ContainerOp, "_DISABLE_REUSABLE_COMPONENT_WARNING"):
        ContainerOp._DISABLE_REUSABLE_COMPONENT_WARNING = True

except ImportError:
    from mlrun_pipelines.common.imports import (
        Compiler,
        PipelineConf,
        PipelineParam,
        compiler,
        dsl,
        kfp,
    )

    ContainerOp = DummyContainerOp
    dsl.ContainerOp = DummyContainerOp

__all__ = [
    "Compiler",
    "ContainerOp",
    "PipelineConf",
    "PipelineParam",
    "compiler",
    "dsl",
    "kfp",
]
