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
#
from typing import Any

import numpy as np
import pandas as pd
from cloudpickle import load

from mlrun.serving.v2_serving import V2ModelServer


class PickleModelServer(V2ModelServer):
    """
    Model serving class, inheriting the V2ModelServer class for being initialized automatically by the model
    server and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.
    """

    def load(self):
        """
        Load and initialize the model and/or other elements.
        """
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))

    def predict(self, request: dict) -> list:
        """
        Infer the inputs through the model using MLRun's PyTorch interface and return its output. The inferred data will
        be read from the "inputs" key of the request.

        :param request: The request of the model. The input to the model will be read from the "inputs" key.
                        The input value can be either a dictionary with the feature names
                        (only one can be sent so only the first one will take into consideration)
                        or a list with feature values.
                        For example, a batch size of 2 for data of two features 'x' and 'y' can be given:

                        * As a dictionary: `{"inputs": [{"x": [1, 2], "y": [3, 5.5]}]}`
                        * As a list: `{"inputs": [[1, 2], [3, 5.5]]}`
        :return: The model's prediction on the given input.
        """
        inputs = request["inputs"]
        if inputs and isinstance(inputs[0], dict):
            x = pd.DataFrame(inputs[0])
        else:
            x = np.asarray(inputs)

        y_pred: np.ndarray = self.model.predict(x)

        return y_pred.tolist()

    def explain(self, request: dict[str, Any]) -> str:
        """
        Returns a string listing the model that is being served in this serving function and the function name.

        :param request: A given request.

        :return: Explanation string.
        """
        return f"A model server named '{self.name}'"
