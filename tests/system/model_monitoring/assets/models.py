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
import mlrun.serving


class OneToOne(mlrun.serving.V2ModelServer):
    """
    In this class the predict method returns one result to each input
    """

    def load(self):
        pass

    def predict(self, body: dict) -> list:
        inputs = body.get("inputs")
        if (
            isinstance(inputs[0], list)
            and isinstance(inputs[0][0], list)
            and len(inputs[0]) == 600
            and len(inputs) == 1
        ):  # single image
            outputs = [3]
        elif isinstance(inputs[0], list) and len(inputs) == 2 and len(inputs[0]) == 600:
            outputs = [2, 2]
        elif isinstance(inputs[0], list):
            outputs = [inp[0] for inp in inputs]
        else:
            outputs = [inputs[0]]
        return outputs


class OneToMany(mlrun.serving.V2ModelServer):
    """
    In this class the predict method returns 5 port outputs result to each input
    """

    def load(self):
        pass

    def predict(self, body: dict) -> list:
        inputs = body.get("inputs")
        if isinstance(inputs[0], list) and len(inputs) > 1:
            outputs = [[inp[0], inp[0], 3.0, "a", 5] for inp in inputs]
        else:
            outputs = [inputs[0], inputs[0], 3.0, "a", 5]
        return outputs
