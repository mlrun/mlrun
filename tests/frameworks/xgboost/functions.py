import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

import mlrun
from mlrun.frameworks._ml_common import AlgorithmFunctionality
from mlrun.frameworks.xgboost import XGBoostTypes, apply_mlrun

from ..ml_functions import MLFunctions


class XGBoostFunctions(MLFunctions):
    @staticmethod
    def train(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_name: str = None
    ):
        algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
        model = XGBoostFunctions.get_model(
            algorithm_functionality=algorithm_functionality
        )
        x_train, x_test, y_train, y_test = XGBoostFunctions.get_dataset(
            algorithm_functionality=algorithm_functionality, for_training=True
        )

        apply_mlrun(model=model, model_name=model_name, x_test=x_test, y_test=y_test)
        model.fit(x_train, y_train)

    @staticmethod
    def evaluate(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_path: str
    ):
        algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
        x, y = XGBoostFunctions.get_dataset(
            algorithm_functionality=algorithm_functionality, for_training=False
        )
        model_handler = apply_mlrun(model_path=model_path, y_test=y)
        model = model_handler.model
        model.predict(x)

    @staticmethod
    def get_model(
        algorithm_functionality: AlgorithmFunctionality,
    ) -> XGBoostTypes.ModelType:
        if algorithm_functionality.is_classification():
            if algorithm_functionality.is_single_output():
                return xgb.XGBClassifier()
            if algorithm_functionality.is_binary_classification():
                return MultiOutputClassifier(xgb.XGBClassifier())
            return MultiOutputClassifier(OneVsRestClassifier(xgb.XGBClassifier()))
        if algorithm_functionality.is_single_output():
            return xgb.XGBRegressor()
        return MultiOutputRegressor(xgb.XGBRegressor())
