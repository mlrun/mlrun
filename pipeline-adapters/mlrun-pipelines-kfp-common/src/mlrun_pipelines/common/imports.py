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
from types import ModuleType
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Define a generic type variable for decorators
Decoratable = TypeVar("Decoratable", bound=Callable[..., Any])


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


@dataclass
class DummyPipelineParam:
    name: str
    value: Any = None


@dataclass
class DummyPipelineConf:
    enable_caching: bool = True
    retries: int = 0

    def set_timeout(self, timeout: int) -> None:
        logger.debug(f"[NoOp] set_timeout called with timeout={timeout}")

    def set_ttl_seconds_after_finished(self, ttl_seconds: int) -> None:
        logger.debug(
            f"[NoOp] set_ttl_seconds_after_finished called with ttl_seconds={ttl_seconds}"
        )

    def add_op_transformer(self, transformer: Callable[[Any], Any]) -> None:
        logger.debug(f"[NoOp] add_op_transformer called with transformer={transformer}")


@dataclass
class DummyPipelineDecorator:
    name: Optional[str] = None
    description: Optional[str] = None

    def __call__(self, func: Decoratable) -> Decoratable:
        logger.debug(f"[NoOp] Pipeline function '{func.__name__}' defined.")
        return func


class DummyContainerOpModule:
    _register_op_handler: Callable[[Any], Any] = lambda x: None


@dataclass
class DummyDSL:
    pipeline: DummyPipelineDecorator = field(default_factory=DummyPipelineDecorator)
    _container_op: DummyContainerOpModule = field(
        default_factory=DummyContainerOpModule
    )

    PipelineParam = DummyPipelineParam
    PipelineConf = DummyPipelineConf


class DummyCompiler:
    @dataclass
    class Compiler:
        def compile(self, pipeline_func: Callable[..., Any], package_path: str) -> None:
            logger.debug(
                f"[NoOp] Compiling pipeline to func '{pipeline_func}' '{package_path}'"
            )

        def _create_workflow(self, *args: Any, **kwargs: Any) -> None:
            logger.debug("[NoOp] _create_workflow called.")


class DummyRunPipelineResult:
    def get_output_file(self, op_name: str, output: Optional[str] = None) -> str:
        return ""

    def success(self) -> bool:
        return True


class V1ListRunsResponse:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @property
    def runs(self):
        return []

    @property
    def next_page_token(self):
        return ""


class DummyClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def create_run_from_pipeline_func(
        self,
        pipeline_func: Callable[..., Any],
        arguments: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "DummyRunPipelineResult":
        logger.debug("[NoOp] create_run_from_pipeline_func called but does nothing.")
        return DummyRunPipelineResult()

    def list_runs(
        self,
        page_token: str = "",
        page_size: int = 100,
        sort_by: Optional[str] = None,
        filter: Optional[str] = None,
    ) -> list[Any]:
        logger.debug("[NoOp] list_runs called")
        return V1ListRunsResponse()


# Assign dummy implementations to kfp modules
compiler = ModuleType("compiler")
compiler.Compiler = DummyCompiler.Compiler()
dsl = ModuleType("dsl")
dsl.PipelineParam = DummyPipelineParam
dsl.PipelineConf = DummyPipelineConf
kfp = ModuleType("kfp")
kfp.compiler = compiler
kfp.dsl = DummyDSL()
ContainerOp = DummyContainerOp
Client = DummyClient
PipelineParam = DummyPipelineParam
PipelineConf = DummyPipelineConf


__all__ = [
    "kfp",
    "dsl",
    "compiler",
    "Client",
    "ContainerOp",
    "DummyClient",
    "DummyCompiler",
    "DummyContainer",
    "DummyContainerOp",
    "DummyDSL",
    "DummyPipelineConf",
    "DummyPipelineDecorator",
    "DummyPipelineParam",
    "PipelineConf",
    "PipelineParam",
]
