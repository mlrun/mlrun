from abc import ABC
from typing import List

import numpy as np
import pandas as pd

from .._common import MLRunInterface, ModelType
from .logger import MLLogger
from .model_handler import MLModelHandler
from .plots import eval_model_v2


class MLMLRunInterface(MLRunInterface, ABC):
    """
    Wraps the original .fit() method of the passed model enabling auto-logging.
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-ml"

    # Properties attributes to be inserted so the ml mlrun interface will be fully enabled:
    _PROPERTIES = {
        "_logger": None,  # type: MLLogger
        "_x_test": None,  # type: MLModelHandler.IOSample
        "_y_test": None,  # type: MLModelHandler.IOSample
    }

    # Methods attributes to be inserted so the ml mlrun interface will be fully enabled:
    _METHODS = ["auto_log", "_pre_fit", "_post_fit", "_pre_predict", "_post_predict"]

    @classmethod
    def add_interface(cls, model: ModelType):
        """
        Wrap the given model with MLRun model features, providing it with MLRun model attributes including its
        parameters and methods.

        :param model: The model to wrap.
        """
        # Wrap the fit method:
        def fit_wrapper(fit_method, **kwargs):
            def wrapper(*args, **kwargs):
                model._pre_fit()
                # Call the original fit method
                fit_method(*args, **kwargs)

                # Original fit method
                setattr(model, "fit", fit_method)

                # Post fit
                model._post_fit(*args, **kwargs)

            return wrapper

        setattr(model, "fit", fit_wrapper(fit_method=model.fit))

    def _post_fit(self, **kwargs):
        eval_metrics = None
        context.set_label("class", str(model.__class__.__name__))

        # Get passed X,y from model.fit(X,y)
        x, y = args[0], args[1]
        # np.array -> Dataframe
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        # Merge X and y for logging of the train set
        train_set = pd.concat([x, y], axis=1)
        train_set.reset_index(drop=True, inplace=True)

        if data.get("X_test") is not None and data.get("y_test") is not None:
            # Identify splits and build test set
            x_test, y_test = data["X_test"], data["y_test"]

            # Merge X and y for logging of the test set
            test_set = pd.concat([x_test, y_test], axis=1)
            test_set.reset_index(drop=True, inplace=True)

            # Evaluate model results and get the evaluation metrics
            eval_metrics = eval_model_v2(context, x_test, y_test, model)

            if data.get("generate_test_set"):
                # Log test dataset
                context.log_dataset(
                    "test_set",
                    df=test_set,
                    format="parquet",
                    index=False,
                    labels={"data-type": "held-out"},
                    artifact_path=context.artifact_subpath("data"),
                )

        # Identify label column
        label_column = None  # type: List[str]
        if isinstance(y, pd.DataFrame):
            label_column = y.columns.to_list()
        elif isinstance(y, pd.Series):
            if y.name is not None:
                label_column = [str(y.name)]
            else:
                raise ValueError("No column name for y was specified")

        model_handler.log(
            algorithm=str(model.__class__.__name__),
            training_set=train_set,
            label_column=label_column,
            extra_data=eval_metrics,
            artifacts=eval_metrics,
            metrics=context.results,
        )
