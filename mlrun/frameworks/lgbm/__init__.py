"""
Description:
__init__ function of LightGBM-autologger. Will be extended
and contain multiple LightGBM-specific functions.
"""

from typing import Union

import numpy as np
import pandas as pd

import mlrun
from mlrun.frameworks._common.pkl_model_server import PickleModelServer
from mlrun.frameworks.mlbase.mlrun_interface import MLBaseMLRunInterface

# Temporary placeholder, LGBMModelServer may
# deviate from PickleModelServer in upcoming versions.
LGBMModelServer = PickleModelServer


def apply_mlrun(
    model,
    context: mlrun.MLClientCtx = None,
    X_test: Union[np.ndarray, pd.core.frame.DataFrame] = None,
    y_test: Union[np.ndarray, pd.core.frame.DataFrame] = None,
    model_name: str = None,
    generate_test_set: bool = True,
    **kwargs
):
    """
    Wrap the given model with MLRun model, saving the model's
    attributes and methods while giving it mlrun's additional features.
    examples::
        model = LGBMClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            test_size=0.2)
        model = apply_mlrun(model, context, X_test=X_test, y_test=y_test)
        model.fit(X_train, y_train)
    :param model:               The model which will have the fit()
                                function wrapped
    :param context:             MLRun context to work with. If no context
                                is given it will be retrieved via
                                'mlrun.get_or_create_ctx(None)'
    :param X_test:              X_test dataset
    :param y_test:              y_test dataset
    :param model_name:          The model artifact name (Optional)
    :param generate_test_set:   Generates a test_set dataset artifact
    :return:                    The model with MLRun's interface.
    """
    if context is None:
        context = mlrun.get_or_create_ctx("mlrun_lgbm")

    kwargs["X_test"] = X_test
    kwargs["y_test"] = y_test
    kwargs["generate_test_set"] = generate_test_set

    # Add MLRun's interface to the model:
    MLBaseMLRunInterface.add_interface(model, context, model_name, kwargs)
    return model
