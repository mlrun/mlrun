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
from typing import Any

import storey

from mlrun.serving import Model, ModelRunnerSelector


class Echo(storey.MapClass):
    def do(self, x):
        print("Echo:", self.name, x)
        return x


class MySelector(ModelRunnerSelector):
    def __init__(self, tool_a, tool_b):
        super().__init__()
        self.tool_a = tool_a
        self.tool_b = tool_b

    def select_outlets(
        self,
        event: Any,
    ) -> list[str] | None:
        count = event.get("counter", 0)
        if count < 3:
            return [self.tool_a]
        elif count < 5:
            return [self.tool_b]
        else:
            return ["end"]


class Tool(storey.MapClass):
    def do(self, event: dict) -> dict:
        event[self.name] = event.get(self.name, 0) + 1
        return event


class LLModelWithTools(Model):
    def load(self):
        pass

    def predict(self, body: Any, **kwargs) -> Any:
        body["counter"] += 1
        return body
