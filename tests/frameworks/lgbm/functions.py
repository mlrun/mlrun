import lightgbm as lgb

import mlrun
from mlrun.frameworks._ml_common import AlgorithmFunctionality
from mlrun.frameworks.lgbm import LGBMMLRunInterface, apply_mlrun

from ..ml_common import N_CLASSES, get_dataset


def get_model(
    algorithm_functionality: AlgorithmFunctionality, n_classes: int = N_CLASSES
) -> lgb.LGBMModel:
    if algorithm_functionality.is_classification():
        if algorithm_functionality.is_single_output():
            return lgb.LGBMClassifier()
        if algorithm_functionality.is_binary_classification():
            return lgb.LGBMClassifier(
                params={"objective": "multiclass", "num_class": 2}
            )
        return lgb.LGBMClassifier(
            params={"objective": "multiclass", "num_class": n_classes}
        )
    return lgb.LGBMRegressor()


def get_params(
    algorithm_functionality: AlgorithmFunctionality, n_classes: int = N_CLASSES
) -> dict:
    if algorithm_functionality.is_classification():
        if algorithm_functionality.is_single_output():
            return {"objective": "binary"}
        if algorithm_functionality.is_binary_classification():
            return {"objective": "multiclass", "num_class": 2}
        return {"objective": "multiclass", "num_class": n_classes}
    return {"objective": "regression"}


class LightGBMHandlers:
    @staticmethod
    def training_api_train(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_name: str = None
    ):
        algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
        params = get_params(algorithm_functionality=algorithm_functionality)
        x_train, x_test, y_train, y_test = get_dataset(
            algorithm_functionality=algorithm_functionality, for_training=True
        )
        train_set = lgb.Dataset(x_train, y_train)
        validation_set_1 = lgb.Dataset(
            x_test[: len(x_test) // 2], y_test[: len(x_test) // 2]
        )
        validation_set_2 = lgb.Dataset(
            x_test[len(x_test) // 2 :], y_test[len(x_test) // 2 :]
        )
        apply_mlrun(model_name=model_name)

        lgb.train(
            params=params,
            train_set=train_set,
            valid_sets=[validation_set_1, validation_set_2],
        )

        # Remove the interface for next test to start from scratch:
        LGBMMLRunInterface.remove_interface(obj=lgb)

    @staticmethod
    def training_api_evaluate(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_path: str
    ):
        # TODO: Finish handler once the evaluation is implemented.
        pass

    @staticmethod
    def sklearn_api_train(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_name: str = None
    ):
        algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
        model = get_model(algorithm_functionality=algorithm_functionality)
        x_train, x_test, y_train, y_test = get_dataset(
            algorithm_functionality=algorithm_functionality, for_training=True
        )

        apply_mlrun(model=model, model_name=model_name, x_test=x_test, y_test=y_test)
        model.fit(x_train, y_train)

    @staticmethod
    def sklearn_api_evaluate(
        context: mlrun.MLClientCtx, algorithm_functionality: str, model_path: str
    ):
        algorithm_functionality = AlgorithmFunctionality(algorithm_functionality)
        x, y = get_dataset(
            algorithm_functionality=algorithm_functionality, for_training=False
        )
        model_handler = apply_mlrun(model_path=model_path, y_test=y)
        model = model_handler.model
        model.predict(x)
