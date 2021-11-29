"""
Description:
__init__ function of Xgboost-autologger. Will be extended and contain multiple Xgboost-specific functions.
"""

import mlrun

from .._ml_common import MLMLRunInterface
from .._ml_common.pkl_model_server import PickleModelServer
from .model_handler import XGBoostModelHandler

# Temporary placeholder, XGBModelServer may deviate from PklModelServer in upcoming versions.
XGBModelServer = PickleModelServer


def apply_mlrun(
    model,
    context: mlrun.MLClientCtx = None,
    X_test=None,
    y_test=None,
    model_name=None,
    generate_test_set=True,
    **kwargs
) -> XGBoostModelHandler:
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.
    examples::

        model = XGBRegressor()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        apply_mlrun(model, context, X_test=X_test, y_test=y_test)
        model.fit(X_train, y_train)

    :param model:       The model to wrap.
    :param context:     MLRun context to work with. If no context is given it will be retrieved via
                        'mlrun.get_or_create_ctx(None)'
    :param X_test:      X test data (for accuracy and plots generation)
    :param y_test:      y test data (for accuracy and plots generation)
    :param model_name:  model artifact name
    :param generate_test_set:  will generate a test_set dataset artifact

    :return: The model with MLRun's interface.
    """
    if context is None:
        context = mlrun.get_or_create_ctx("mlrun_xgb")

    kwargs["X_test"] = X_test
    kwargs["y_test"] = y_test
    kwargs["generate_test_set"] = generate_test_set

    mh = XGBoostModelHandler(
        model_name=model_name or "model", model=model, context=context
    )

    # Add MLRun's interface to the model:
    MLMLRunInterface.add_interface(mh, context, model_name, kwargs)
    return mh
