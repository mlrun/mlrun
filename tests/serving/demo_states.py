# Copyright 2023 Iguazio
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

from collections.abc import Sequence
from copy import copy
from typing import Any, Optional

from mlrun.serving import Model, ModelRunnerSelector, V2ModelServer


class BaseClass:
    def __init__(self, context, name=None):
        self.context = context
        self.name = name


class Echo(BaseClass):
    def __init__(self, name=None):
        self.name = name

    def do(self, x):
        print("Echo:", self.name, x)
        return x


class RespName(BaseClass):
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")

    def do(self, x):
        print("Echo:", self.name, x)
        return [x, self.name]


class EchoError(BaseClass):
    def do(self, x):
        x.body = {"body": x.body, "origin_state": x.origin_state, "error": x.error}
        print("EchoError:", x)
        return x


class Chain(BaseClass):
    def do(self, x):
        x = copy(x)
        x.append(self.name)
        return x


class ChainWithContext(BaseClass):
    def do(self, x):
        visits = self.context.visits.get(self.name, 0)
        self.context.visits[self.name] = visits + 1
        x = copy(x)
        x.append(self.name)
        return x


class Message(BaseClass):
    def __init__(self, msg="", context=None, name=None):
        self.msg = msg

    def do(self, x):
        print("Messsage:", self.msg)
        return x


class Raiser:
    def __init__(self, msg="", context=None, name=None):
        self.context = context
        self.name = name
        self.msg = msg

    def do(self, x):
        raise ValueError(f" this is an error, {x}")


def multiply_input(request):
    request["inputs"][0] = request["inputs"][0] * 2
    return request


class ModelClass(V2ModelServer):
    def load(self):
        print("loading")

    def predict(self, request):
        print("predict:", request)
        resp = request["inputs"][0] * self.get_param("multiplier", 1)
        return resp


class ModelClassList(V2ModelServer):
    def load(self):
        print("loading")

    def predict(self, request):
        print("predict:", request)
        resp = request["inputs"][0][0] * self.get_param("multiplier", 1)
        return [resp]


class Route:
    def do(self, event):
        print("Before routing", event)
        return event

    def select_outlets(self, event):
        if event.get("go_cyclic"):
            return ["count"]
        return ["end"]


class Counter:
    def do(self, event: dict):
        event["counter"] = event.get("counter", 0) + 1
        event["go_cyclic"] = True
        if event["counter"] > 4:
            event["go_cyclic"] = False
        return event


class LLModelWithTools(Model):
    def load(self):
        pass

    def predict(self, body: Any, **kwargs) -> Any:
        body["counter"] += 1
        return body


class MySelector(ModelRunnerSelector):
    def select_outlets(
        self,
        event: Any,
    ) -> Optional[Sequence[str]]:
        count = event.get("counter", 0)
        if count < 3:
            return ["tool_a"]
        elif count < 5:
            return ["tool_b"]
        else:
            return ["end"]


class Tool(BaseClass):
    def do(self, event: dict) -> dict:
        event[self.name] = event.get(self.name, 0) + 1
        return event


class MyRemoteModel(Model):
    def predict(self, body, **kwargs):
        body["url"] = self.model_artifact.model_url
        body["default_config"] = self.model_artifact.default_config
        return body

    async def predict_async(self, body, **kwargs):
        body["async_triggered"] = "Async predict was triggered."
        return body
