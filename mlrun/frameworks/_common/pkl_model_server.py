import numpy as np
from cloudpickle import load
from sklearn.base import is_regressor

from mlrun.serving.v2_serving import V2ModelServer


class PickleModelServer(V2ModelServer):
    """
    Model serving class, inheriting the V2ModelServer class for being initialized automatically by the model
    server and be able to run locally as part of a nuclio serverless function, or as part of a real-time pipeline.
    """

    def load(self):
        """
        Use the model handler to load the model.
        """
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))

    def predict(self, body: dict) -> list:
        """
        Infer the inputs through the model using MLRun's PyTorch interface and return its output. The inferred data will
        be read from the "inputs" key of the request.
        :param request: The request of the model. The input to the model will be read from the "inputs" key.
        :return: The model's prediction on the given input.
        """
        feats = np.asarray(body["inputs"])

        if is_regressor(self.model):
            result: np.ndarray = self.model.score(feats)
        else:
            result: np.ndarray = self.model.predict(feats)
        return result.tolist()
