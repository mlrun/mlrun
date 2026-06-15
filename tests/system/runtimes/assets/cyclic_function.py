# Copyright 2025 Iguazio
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
import storey


class Echo(storey.MapClass):
    def do(self, event):
        print("Echo:", self.name, event)
        return event


class Route:
    def __init__(self, end="end"):
        self.end = end

    def do(self, event):
        print("Before routing", event)
        return event

    def select_outlets(self, event):
        if event.get("go_cyclic"):
            return ["count"]
        return [self.end]


class Counter:
    def do(self, event: dict):
        event["counter"] = event.get("counter", 0) + 1
        event["go_cyclic"] = True
        if event["counter"] > 4:
            event["go_cyclic"] = False
        return event
