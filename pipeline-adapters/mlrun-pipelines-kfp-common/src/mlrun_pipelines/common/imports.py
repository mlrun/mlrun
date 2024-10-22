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
logger.addHandler(logging.NullHandler())

# Define a generic type variable for decorators
Decoratable = TypeVar("Decoratable", bound=Callable[..., Any])

# Initialize placeholders for KFP components
kfp: ModuleType = ModuleType("kfp")
dsl: ModuleType = ModuleType("dsl")
compiler: ModuleType = ModuleType("compiler")
ContainerOp: Optional[type["ContainerOpBase"]] = None
Client: Optional[type["ClientBase"]] = None
PipelineParam: Optional[type["PipelineParamBase"]] = None
PipelineConf: Optional[type["PipelineConfBase"]] = None


class DummyContainer:
    def __init__(self) -> None:
        self.env: list[dict[str, str]] = []  # list of environment variables
        self.command: list[str] = []  # Command to run in the container
        self.args: list[str] = []  # Arguments for the command
        self.image: str = ""  # Container image
        self.resources: dict[str, Any] = {}  # Resource limits/requests

    def set_command(self, command: list[str]) -> None:
        logger.warning(f"[NoOp] set_command called with command={command}")
        self.command = command

    def set_args(self, args: list[str]) -> None:
        logger.warning(f"[NoOp] set_args called with args={args}")
        self.args = args

    def set_image(self, image: str) -> None:
        logger.warning(f"[NoOp] set_image called with image={image}")
        self.image = image

    def add_env_variable(self, name: str, value: str) -> None:
        logger.warning(
            f"[NoOp] add_env_variable called with name='{name}', value='{value}'"
        )
        self.env.append({"name": name, "value": value})

    def set_resources(self, resources: dict[str, Any]) -> None:
        logger.warning(f"[NoOp] set_resources called with resources={resources}")
        self.resources = resources


class ContainerOpBase:
    name: str
    image: str
    command: list
    file_outputs: dict[str, str]
    kwargs: dict[str, Any]
    pod_labels: dict[str, str]
    pod_annotations: dict[str, str]
    volumes: list[Any]
    container: DummyContainer

    def add_pod_label(self, key: str, value: str) -> None:
        pass

    def add_volume(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_env_variable(self, name: str, value: str) -> None:
        pass


class DummyContainerOp(ContainerOpBase):
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
        logger.warning(
            f"[NoOp] ContainerOp created with name='{name}', image='{image}', "
            f"command={command}, file_outputs={self.file_outputs}, kwargs={self.kwargs}"
        )

    def add_pod_label(self, key: str, value: str) -> None:
        logger.warning(f"[NoOp] add_pod_label called with key='{key}', value='{value}'")
        self.pod_labels[key] = value

    def add_volume(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(f"[NoOp] add_volume called with args={args}, kwargs={kwargs}")
        self.volumes.append({"args": args, "kwargs": kwargs})

    def add_env_variable(self, name: str, value: str) -> None:
        logger.warning(
            f"[NoOp] add_env_variable called with name='{name}', value='{value}'"
        )
        self.container.env.append({"name": name, "value": value})


class PipelineParamBase:
    name: str
    value: Any

    def __init__(self, name: str, value: Any = None) -> None:
        self.name = name
        self.value = value
        logger.warning(
            f"[NoOp] PipelineParam created with name='{name}', value='{value}'"
        )


class DummyPipelineParam(PipelineParamBase):
    def __init__(self, name: str, value: Any = None) -> None:
        super().__init__(name, value)


class PipelineConfBase:
    enable_caching: bool
    retries: int

    def __init__(self, enable_caching: bool = True, retries: int = 0) -> None:
        self.enable_caching = enable_caching
        self.retries = retries
        logger.warning(
            f"[NoOp] PipelineConf created with enable_caching={enable_caching}, retries={retries}"
        )

    def set_timeout(self, timeout: int) -> None:
        logger.warning(f"[NoOp] set_timeout called with timeout={timeout}")

    def set_ttl_seconds_after_finished(self, ttl_seconds: int) -> None:
        logger.warning(
            f"[NoOp] set_ttl_seconds_after_finished called with ttl_seconds={ttl_seconds}"
        )

    def add_op_transformer(self, transformer: Callable[[Any], Any]) -> None:
        logger.warning(
            f"[NoOp] add_op_transformer called with transformer={transformer}"
        )


class DummyPipelineConf(PipelineConfBase):
    def __init__(self, enable_caching: bool = True, retries: int = 0) -> None:
        super().__init__(enable_caching, retries)

    def set_timeout(self, timeout: int) -> None:
        super().set_timeout(timeout)

    def set_ttl_seconds_after_finished(self, ttl_seconds: int) -> None:
        super().set_ttl_seconds_after_finished(ttl_seconds)

    def add_op_transformer(self, transformer: Callable[[Any], Any]) -> None:
        super().add_op_transformer(transformer)


class DummyPipelineDecorator:
    def __init__(
        self, name: Optional[str] = None, description: Optional[str] = None
    ) -> None:
        self.name = name
        self.description = description
        logger.warning(
            f"[NoOp] Pipeline decorator used: name='{name}', description='{description}'"
        )

    def __call__(self, func: Decoratable) -> Decoratable:
        logger.warning(f"[NoOp] Pipeline function '{func.__name__}' defined.")
        return func


class DummyContainerOpModule:
    _register_op_handler: Callable[[Any], Any] = lambda x: None


class DummyDSL:
    pipeline: Callable[[Optional[str], Optional[str]], DummyPipelineDecorator]

    def __init__(self) -> None:
        self.pipeline = DummyPipelineDecorator()
        self._container_op = DummyContainerOpModule()

    PipelineParam: type[PipelineParamBase] = DummyPipelineParam
    PipelineConf: type[PipelineConfBase] = DummyPipelineConf


class DummyCompiler:
    class Compiler:
        def compile(self, pipeline_func: Callable[..., Any], package_path: str) -> None:
            logger.warning(
                f"[NoOp] Compiling pipeline to '{package_path}'. This is a no-operation because KFP is not installed."
            )


class ClientBase:
    def create_run_from_pipeline_func(
        self,
        pipeline_func: Callable[..., Any],
        arguments: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        pass


class DummyClient(ClientBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning(
            "[NoOp] KFP Client initialized but will not perform any operations."
        )

    def create_run_from_pipeline_func(
        self,
        pipeline_func: Callable[..., Any],
        arguments: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        logger.warning("[NoOp] create_run_from_pipeline_func called but does nothing.")


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
