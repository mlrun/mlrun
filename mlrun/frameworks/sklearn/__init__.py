"""
Description:
__init__ function of sklearn-autologger. Will be extended and contain multiple Sklearn-specific functions.
"""

import mlrun

from .._ml_common.mlrun_interface import MLMLRunInterface
from .._ml_common.pkl_model_server import PickleModelServer
from .model_handler import SKLearnModelHandler

# Temporary placeholder, SklearnModelServer may deviate from PklModelServer in upcoming versions.
SklearnModelServer = PickleModelServer


def apply_mlrun(
    model,
    context: mlrun.MLClientCtx = None,
    X_test=None,
    y_test=None,
    model_name=None,
    tag: str = "",
    generate_test_set=True,
    **kwargs
) -> SKLearnModelHandler:
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.

    examples:
        model = LogisticRegression()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = apply_mlrun(model, context, X_test=X_test, y_test=y_test)
        model.fit(X_train, y_train)

    :param model:             The model which will have the fit() function wrapped
    :param context:           MLRun context to work with. If no context is given it will be retrieved via
                              'mlrun.get_or_create_ctx(None)'
    :param X_test:            X_test dataset
    :param y_test:            y_test dataset
    :param model_name:        The model artifact name (Optional)
    :param tag:               Tag of a version to give to the logged model.
    :param generate_test_set: Generates a test_set dataset artifact

    :return: The model in a MLRun model handler.
    """
    if context is None:
        context = mlrun.get_or_create_ctx("mlrun_sklearn")

    kwargs["X_test"] = X_test
    kwargs["y_test"] = y_test
    kwargs["generate_test_set"] = generate_test_set

    mh = SKLearnModelHandler(
        model_name=model_name or "model", model=model, context=context
    )

    # Add MLRun's interface to the model:
    MLMLRunInterface.add_interface(mh, context, tag, kwargs)
    return mh
