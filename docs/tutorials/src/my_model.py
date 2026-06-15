import numpy as np
import pandas as pd
from cloudpickle import load

from mlrun.serving import Model


class MyModel(Model):
    def __init__(
        self, *args, artifact_uri: str | None = None, raise_exception=False, **kwargs
    ):
        super().__init__(
            *args, artifact_uri=artifact_uri, raise_exception=raise_exception, **kwargs
        )

    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, _ = self.get_local_model_path(".pkl")
        self.model = load(open(model_file, "rb"))

    def predict(self, data: dict, **kwargs) -> dict:
        """Generate model predictions from sample."""
        df = pd.DataFrame([data])
        y_pred: np.ndarray = self.model.predict(df)
        data["label"] = int(y_pred[0])
        return data
