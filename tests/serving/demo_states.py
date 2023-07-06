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
#
from copy import copy

from mlrun.serving import V2ModelServer


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
