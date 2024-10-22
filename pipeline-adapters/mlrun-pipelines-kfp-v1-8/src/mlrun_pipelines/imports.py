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

logger = logging.getLogger(__name__)

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
    ContainerOp = real_ContainerOp
    Client = real_Client
    PipelineParam = real_PipelineParam
    PipelineConf = real_PipelineConf

    if hasattr(ContainerOp, "_DISABLE_REUSABLE_COMPONENT_WARNING"):
        ContainerOp._DISABLE_REUSABLE_COMPONENT_WARNING = True

except ImportError:
    logger.warning(
        "Kubeflow Pipelines (KFP) is not installed. Using no-operation (noop) implementations."
    )
    from mlrun_pipelines.common.imports import (
        DummyClient,
        DummyCompiler,
        DummyContainerOp,
        DummyDSL,
        DummyPipelineConf,
        DummyPipelineParam,
    )

    PipelineConf = DummyPipelineConf
    PipelineParam = DummyPipelineParam
    ContainerOp = DummyContainerOp
    Client = DummyClient
    dsl = DummyDSL()
    compiler = DummyCompiler()


__all__ = [
    "kfp",
    "dsl",
    "compiler",
    "ContainerOp",
    "Client",
    "PipelineParam",
    "PipelineConf",
]
