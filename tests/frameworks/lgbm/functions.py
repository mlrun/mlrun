import lightgbm as lgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

import mlrun
import mlrun.frameworks.lgbm as mlrun_lgbm
from mlrun.frameworks._ml_common.utils import AlgorithmFunctionality, ModelType

from ..functions import MLFunctions


class LGBMFunctions(MLFunctions):
    @staticmethod
    def train(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_name: str = None
    ):
        algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
        model = LGBMFunctions._get_model(
            algorithm_functionality=algorithm_functionality
        )
        x_train, x_test, y_train, y_test = LGBMFunctions._get_dataset(
            algorithm_functionality=algorithm_functionality, for_training=True
        )

        mlrun_lgbm.apply_mlrun(
            model=model, model_name=model_name, x_test=x_test, y_test=y_test
        )
        model.fit(x_train, y_train)

    @staticmethod
    def evaluate(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_path: str
    ):
        algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
        x, y = LGBMFunctions._get_dataset(
            algorithm_functionality=algorithm_functionality, for_training=False
        )
        model_handler = mlrun_lgbm.apply_mlrun(model_path=model_path, y_test=y)
        model = model_handler.model
        model.predict(x)

    @staticmethod
    def _get_model(algorithm_functionality: AlgorithmFunctionality) -> ModelType:
        if algorithm_functionality.is_classification():
            if algorithm_functionality.is_single_output():
                return lgb.LGBMClassifier()
            if algorithm_functionality.is_binary_classification():
                return MultiOutputClassifier(lgb.LGBMClassifier())
            return MultiOutputClassifier(OneVsRestClassifier(lgb.LGBMClassifier()))
        if algorithm_functionality.is_single_output():
            return lgb.LGBMRegressor()
        return MultiOutputRegressor(lgb.LGBMRegressor())
