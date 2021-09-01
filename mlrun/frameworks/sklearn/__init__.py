"""
Description:
__init__ function of sklearn-autologger. Will be extended and contain multiple Sklearn-specific functions.
"""

import mlrun
from mlrun.frameworks._common.pkl_model_server import PickleModelServer
from mlrun.frameworks.mlbase.mlrun_interface import MLBaseMLRunInterface

# Temporary placeholder, SklearnModelServer may deviate from PklModelServer in upcoming versions.
SklearnModelServer = PickleModelServer


def apply_mlrun(model, context: mlrun.MLClientCtx = None, model_name=None, **kwargs):
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.

    examples::          model = LogisticRegression()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                        model = apply_mlrun_xgb(model, context, X_test=X_test, y_test=y_test)
                        model.fit(X_train, y_train)

    examples::          model = LogisticRegression()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                        model = apply_mlrun_xgb(model, context)
                        model.fit(X_train, y_train)

    :param model:       The model to wrap.

    :param context:     MLRun context to work with. If no context is given it will be retrieved via
                        'mlrun.get_or_create_ctx(None)'

    :return: The model with MLRun's interface.
    """
    if context is None:
        context = mlrun.get_or_create_ctx("mlrun_sklearn")

    # Add MLRun's interface to the model:
    MLBaseMLRunInterface.add_interface(model, context, model_name, kwargs)
    return model
