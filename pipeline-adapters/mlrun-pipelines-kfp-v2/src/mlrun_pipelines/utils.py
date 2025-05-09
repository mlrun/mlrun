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

import tempfile
import typing

import mlrun_pipelines.imports


def compile_pipeline(
    pipeline, pipe_file: typing.Optional[str] = None, type_check: bool = False, **kwargs
):
    if not pipe_file:
        pipe_file = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    mlrun_pipelines.imports.Compiler().compile(
        pipeline, pipe_file, type_check=type_check
    )
    return pipe_file


def get_client(
    url: typing.Optional[str] = None, namespace: typing.Optional[str] = None
) -> mlrun_pipelines.imports.kfp.Client:
    if url or namespace:
        return mlrun_pipelines.imports.kfp.Client(host=url, namespace=namespace)
    return mlrun_pipelines.imports.kfp.Client()
