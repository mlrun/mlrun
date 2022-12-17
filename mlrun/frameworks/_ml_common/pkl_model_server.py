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
import numpy as np
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

    def predict(self, body: dict) -> list:
        """
        Infer the inputs through the model using MLRun's PyTorch interface and return its output. The inferred data will
        be read from the "inputs" key of the request.

        :param request: The request of the model. The input to the model will be read from the "inputs" key.

        :return: The model's prediction on the given input.
        """
        x = np.asarray(body["inputs"])

        y_pred: np.ndarray = self.model.predict(x)

        return y_pred.tolist()
