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
import typing

import numpy as np
from cloudpickle import load

import mlrun.artifacts
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


class IncModel(mlrun.serving.states.Model):
    def __init__(
        self, *args, inc: int, gpu_number: typing.Optional[int] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.inc = inc
        self.gpu_number = gpu_number

    def predict(self, body, **kwargs):
        body["n"] += self.inc
        body.pop("models", None)
        if self.gpu_number is not None:
            body["gpu"] = self.gpu_number
        return body

    async def predict_async(self, body):
        return self.predict(body)


class MyRemoteModel(mlrun.serving.states.Model):
    def predict(self, body, **kwargs):
        body["url"] = self.model_artifact.model_url
        body["default_config"] = self.model_artifact.default_config
        return body


class Echo:
    def __init__(self, name=None):
        self.name = name

    def do(self, x):
        print("Echo:", self.name, x)
        return x


class MyModel(mlrun.serving.Model):
    def __init__(
        self,
        *args,
        artifact_uri: typing.Optional[str] = None,
        raise_exception: bool = False,
        gpu_number: typing.Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            *args, artifact_uri=artifact_uri, raise_exception=raise_exception, **kwargs
        )
        self.gpu_number = gpu_number
        self.model_spec = None
        self.model = None
        self._params = {}

    def get_model(self, suffix=""):
        """get the model file(s) and metadata from model store

        the method returns a path to the model file and the extra data (dict of dataitem objects)
        it also loads the model metadata into the self.model_spec attribute, allowing direct access
        to all the model metadata attributes.

        get_model is usually used in the model .load() method to init the model
        Examples
        --------
        ::

            def load(self):
                model_file, extra_data = self.get_model(suffix=".pkl")
                self.model = load(open(model_file, "rb"))
                categories = extra_data["categories"].as_df()

        Parameters
        ----------
        suffix : str
            optional, model file suffix (when the model_path is a directory)

        Returns
        -------
        str
            (local) model file
        dict
            extra dataitems dictionary

        """
        if self.artifact_uri:
            model_file, self.model_spec, extra_dataitems = mlrun.artifacts.get_model(
                self.artifact_uri, suffix
            )
            if self.model_spec and self.model_spec.parameters:
                for key, value in self.model_spec.parameters.items():
                    self._params[key] = value
            return model_file, extra_dataitems
        return None, None

    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))

    def predict(self, body: dict, **kwargs) -> dict:
        """Generate model predictions from sample."""
        feats = np.asarray(body["inputs"])
        start = mlrun.utils.now_date().isoformat(sep=" ", timespec="microseconds")
        result: np.ndarray = self.model.predict(feats)
        body["outputs"] = result.tolist()
        body["timestamp"] = start
        return body

    async def predict_async(self, body):
        return self.predict(body)
