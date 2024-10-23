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
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Define a generic type variable for decorators
Decoratable = TypeVar("Decoratable", bound=Callable[..., Any])

# Initialize placeholders for KFP components
kfp: ModuleType = ModuleType("kfp")
dsl: ModuleType = ModuleType("dsl")
compiler: ModuleType = ModuleType("compiler")
ContainerOp: Optional[type["DummyContainerOp"]] = None
Client: Optional[type["DummyClient"]] = None
PipelineParam: Optional[type["PipelineParam"]] = None
PipelineConf: Optional[type["PipelineConf"]] = None


class DummyContainer:
    def __init__(self) -> None:
        self.env: list[dict[str, str]] = []
        self.command: list[str] = []
        self.args: list[str] = []
        self.image: str = ""
        self.resources: dict[str, Any] = {}

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


class DummyContainerOp:
    def __init__(
        self,
        name: str,
        image: str,
        command: list,
        file_outputs: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.image = image
        self.command = command
        self.file_outputs = file_outputs or {}
        self.kwargs = kwargs
        self.pod_labels = {}
        self.pod_annotations = {}
        self.volumes = []
        self.container = DummyContainer()

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


class DummyPipelineParam:
    def __init__(self, name: str, value: Any = None) -> None:
        self.name = name
        self.value = value


class DummyPipelineConf:
    def __init__(self, enable_caching: bool = True, retries: int = 0) -> None:
        self.enable_caching = enable_caching
        self.retries = retries

    def set_timeout(self, timeout: int) -> None:
        logger.debug(f"[NoOp] set_timeout called with timeout={timeout}")

    def set_ttl_seconds_after_finished(self, ttl_seconds: int) -> None:
        logger.debug(
            f"[NoOp] set_ttl_seconds_after_finished called with ttl_seconds={ttl_seconds}"
        )

    def add_op_transformer(self, transformer: Callable[[Any], Any]) -> None:
        logger.debug(f"[NoOp] add_op_transformer called with transformer={transformer}")


class DummyPipelineDecorator:
    def __init__(
        self, name: Optional[str] = None, description: Optional[str] = None
    ) -> None:
        self.name = name
        self.description = description

    def __call__(self, func: Decoratable) -> Decoratable:
        logger.debug(f"[NoOp] Pipeline function '{func.__name__}' defined.")
        return func


class DummyContainerOpModule:
    _register_op_handler: Callable[[Any], Any] = lambda x: None


class DummyDSL:
    pipeline: Callable[[Optional[str], Optional[str]], DummyPipelineDecorator]

    def __init__(self) -> None:
        self.pipeline = DummyPipelineDecorator()
        self._container_op = DummyContainerOpModule()

    PipelineParam: type[PipelineParam] = DummyPipelineParam
    PipelineConf: type[PipelineConf] = DummyPipelineConf


class DummyCompiler:
    class Compiler:
        def compile(self, pipeline_func: Callable[..., Any], package_path: str) -> None:
            logger.debug(
                f"[NoOp] Compiling pipeline to func '{pipeline_func}' '{package_path}'"
            )

    def _create_workflow(self, *args, **kwargs):
        logger.debug("[NoOp] _create_workflow called.")


class DummyClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def create_run_from_pipeline_func(
        self,
        pipeline_func: Callable[..., Any],
        arguments: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        logger.debug("[NoOp] create_run_from_pipeline_func called but does nothing.")

    def list_runs(
        self,
        page_token: str = "",
        page_size: int = 100,
        sort_by: Optional[str] = None,
        filter: Optional[str] = None,
    ) -> None:
        logger.debug("[NoOp] list_runs called")


compiler.Compiler = DummyCompiler()
kfp.compiler = compiler
dsl.PipelineParam = DummyPipelineParam
dsl.PipelineConf = DummyPipelineConf
kfp.dsl = dsl

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
