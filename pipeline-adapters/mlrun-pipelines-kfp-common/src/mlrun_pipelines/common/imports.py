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
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Define a generic type variable for decorators
Decoratable = TypeVar("Decoratable", bound=Callable[..., Any])


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


class DummyCompiler:
    @dataclass
    class Compiler:
        _has_warned: bool = False

        def _warn_once_about_kfp(self):
            if not self._has_warned:
                logger.warning("KFP is not installed; using a no-op compiler.")
                self._has_warned = True

        def compile(
            self,
            pipeline_func: Optional[Callable[..., Any]] = None,
            package_path: Optional[str] = None,
            **kwargs: Any,
        ) -> None:
            self._warn_once_about_kfp()
            logger.debug(
                f"[NoOp] Compiling pipeline for func '{pipeline_func}' -> '{package_path}'"
            )

        def _create_workflow(self, *args: Any, **kwargs: Any) -> None:
            self._warn_once_about_kfp()
            logger.debug("[NoOp] _create_workflow called.")

        def __call__(self) -> "DummyCompiler.Compiler":
            return self


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
Compiler = DummyCompiler.Compiler()
compiler.Compiler = Compiler
dsl = ModuleType("dsl")
dsl.pipeline = DummyPipelineDecorator
dsl.PipelineParam = DummyPipelineParam
dsl.PipelineConf = DummyPipelineConf
kfp = ModuleType("kfp")
kfp.compiler = compiler
kfp.dsl = dsl
kfp.Client = DummyClient
Client = DummyClient
PipelineParam = DummyPipelineParam
PipelineConf = DummyPipelineConf


__all__ = [
    "Client",
    "Compiler",
    "PipelineConf",
    "PipelineParam",
    "compiler",
    "dsl",
    "kfp",
]
