# Copyright 2026 Iguazio
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

"""Shared helpers for OpenAI endpoint test modules."""

from typing import cast

import mlrun
from mlrun.runtimes.nuclio.serving import ServingRuntime
from mlrun.serving.endpoint_mapping import APIHandlerConfig
from mlrun.serving.openai_mappings import OpenAIEndpoint
from mlrun.serving.server import GraphServer


def make_fn() -> ServingRuntime:
    return cast(ServingRuntime, mlrun.new_function("test-openai", kind="serving"))


def get_config(fn: ServingRuntime) -> APIHandlerConfig:
    return APIHandlerConfig.from_dict(fn.spec.api_handler_config)


def make_mock_server(endpoint_group: OpenAIEndpoint, handler) -> GraphServer:
    fn = make_fn()
    fn.set_openai_frontend([endpoint_group])
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="handler", handler=handler).respond()
    return fn.to_mock_server()
