from typing import List

import numpy as np
import pandas as pd

from .model_handler import MLModelHandler
from .plots import eval_model_v2


class MLMLRunInterface:
    """
    Wraps the original .fit() method of the passed model enabling auto-logging.
    """

    @classmethod
    def add_interface(
        cls, model_handler: MLModelHandler, context, tag, data={}, *args, **kwargs
    ):
        """
        Wrap the given model with MLRun model features, providing it with MLRun model attributes including its
        parameters and methods.

        :param model_handler: The model to wrap.
        :param context:       MLRun context to work with. If no context is given it will be retrieved via
                              'mlrun.get_or_create_ctx(None)'
        :param tag:           Tag for the model to log with.
        :param data:          The train_test_split X_train, X_test, y_train, y_test can be passed, or the test data
                              X_test, y_test can be passed.
        """
        model = model_handler.model

        # Wrap the fit method:
        def fit_wrapper(fit_method, **kwargs):
            def wrapper(*args, **kwargs):
                # Call the original fit method
                fit_method(*args, **kwargs)

                # Original fit method
                setattr(model, "fit", fit_method)

                # Post fit
                _post_fit(*args, **kwargs)

            return wrapper

        setattr(model, "fit", fit_wrapper(model.fit, **kwargs))

        def _post_fit(*args, **kwargs):
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
                tag=tag,
                algorithm=str(model.__class__.__name__),
                training_set=train_set,
                label_column=label_column,
                extra_data=eval_metrics,
                artifacts=eval_metrics,
                metrics=context.results,
            )
