import pandas as pd

from mlrun.frameworks._ml_common.model_handler import MLModelHandler
from mlrun.frameworks._ml_common.plots import eval_model_v2


class MLMLRunInterface:
    """
    Wraps the original .fit() method of the passed model enabling auto-logging.
    """

    @classmethod
    def add_interface(
        cls,
        model_handler: MLModelHandler,
        context,
        model_name,
        data={},
        *args,
        **kwargs
    ):
        """
        Wrap the given model with MLRun model features, providing it with MLRun model attributes including its
        parameters and methods.

        :param model:       The model to wrap.
        :param context:     MLRun context to work with. If no context is given it will be retrieved via
                            'mlrun.get_or_create_ctx(None)'
        :param model_name:  name under whcih the model will be saved within the databse.
        :param data:        Optional: The train_test_split X_train, X_test, y_train, y_test can be passed,
                                      or the test data X_test, y_test can be passed.

        :return: The wrapped model.
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

            # Identify splits and build test set
            X_train = args[0]
            y_train = args[1]
            train_set = pd.concat([X_train, y_train], axis=1)
            train_set.reset_index(drop=True, inplace=True)

            if data.get("X_test") is not None and data.get("y_test") is not None:
                # Identify splits and build test set
                X_test = data["X_test"]
                y_test = data["y_test"]
                test_set = pd.concat([X_test, y_test], axis=1)
                test_set.reset_index(drop=True, inplace=True)

                # Evaluate model results and get the evaluation metrics
                eval_metrics = eval_model_v2(context, X_test, y_test, model)

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

            # Log fitted model and metrics
            label_column = (
                y_train.name
                if isinstance(y_train, pd.Series)
                else y_train.columns.to_list()
            )
            model_handler.log(
                algorithm=str(model.__class__.__name__),
                training_set=train_set,
                label_column=label_column,
                extra_data=eval_metrics,
                artifacts=eval_metrics,
                metrics=context.results,
            )
