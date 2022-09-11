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

from tabulate import tabulate

import mlrun.utils.helpers
from .base import NotificationBase


class ConsoleNotification(NotificationBase):
    def send(self):
        print(f"[{self.severity}] {self.header}")

        if not self.runs:
            return

        table = []
        for run in self.runs:
            state = run["status"].get("state", "")
            if state == "error":
                result = run["status"].get("error", "")
            else:
                result = mlrun.utils.helpers.dict_to_str(
                    run["status"].get("results", {})
                )

            table.append(
                [
                    state,
                    run["metadata"]["name"],
                    ".." + run["metadata"]["uid"][-6:],
                    result,
                ]
            )
        print(tabulate(table, headers=["status", "name", "uid", "results"]))
