# Copyright 2018 Iguazio
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

import mlrun

from .pipelines import pipeline_context


def function_run_step(
    function_uri,
    handler=None,
    name: str = "",
    params: dict = None,
    hyperparams=None,
    selector="",
    hyper_param_options: mlrun.model.HyperParamOptions = None,
    inputs: dict = None,
    outputs: dict = None,
    workdir: str = "",
    artifact_path: str = "",
    image: str = "",
    labels: dict = None,
    verbose=None,
):
    function: mlrun.runtimes.BaseRuntime = pipeline_context.get_function(function_uri)
    name = name or function.metadata.name
    return function.run(
        handler=handler,
        name=name,
        params=params,
        hyperparams=hyperparams,
        hyper_param_options=hyper_param_options,
        inputs=inputs,
        workdir=workdir,
        artifact_path=artifact_path,
        labels=labels,
        verbose=verbose,
        local=True,
    )
