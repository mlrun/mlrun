"""
Description:
__init__ function of LightGBM-autologger. Will be extended and contain multiple LightGBM-specific functions.
"""

import mlrun
import numpy as np
import pandas as pd
import mlrun.feature_store import FeatureVector

from mlrun.frameworks._common.pkl_model_server import PickleModelServer
from mlrun.frameworks.mlbase.mlrun_interface import MLBaseMLRunInterface
from typing import Union

# Temporary placeholder, LGBMModelServer may deviate from PickleModelServer in upcoming versions.
LGBMModelServer = PickleModelServer


def apply_mlrun(
        model,
        context: mlrun.MLClientCtx = None,
        X_test: Union[numpy.ndarray, pd.core.frame.DataFrame] = None,
        y_test: Union[numpy.ndarray, pd.core.frame.DataFrame] = None,
        model_name: str = None,
        generate_test_set: bool = True,
        feature_vector: Union[FeatureVector, str] = None,
        **kwargs
):
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.
    
    examples::
        model = LGBMClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = apply_mlrun(model, context, X_test=X_test, y_test=y_test)
        model.fit(X_train, y_train)
        
    :param model:               The model to wrap.
    :param context:             MLRun context to work with. If no context is given it will be retrieved via 'mlrun.get_or_create_ctx(None)'
    :param X_test:              X test data (for accuracy and plots generation)
    :param y_test:              y test data (for accuracy and plots generation)
    :param model_name:          model artifact name
    :param generate_test_set:   will generate a test_set dataset artifact
    :param feature_vector:      = None,
    
    :return:                    The model with MLRun's interface.
    """
    if context is None:
        context = mlrun.get_or_create_ctx("mlrun_lgbm")

    if feature_vector and hasattr(feature_vector, "uri"):
        kwargs["feature_vector"] = feature_vector.uri
        
    elif feature_vector and isinstance(feature_vector, str):
        kwargs["feature_vector"] = feature_vector
        

    kwargs["X_test"] = X_test
    kwargs["y_test"] = y_test
    kwargs["generate_test_set"] = generate_test_set

    # Add MLRun's interface to the model:
    MLBaseMLRunInterface.add_interface(model, context, model_name, kwargs)
    return model
