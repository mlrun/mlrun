import pandas as pd
from mlrun.frameworks._common.plots import eval_model_v2
from mlrun.frameworks._common import MLRunInterface
from cloudpickle import dumps

class MLBaseMLRunInterface(MLRunInterface):
    """
    Wraps the original .fit() method of the passed model enabling auto-logging.
    """

    @classmethod
    def add_interface(cls, model, context, model_name, data={}, *args, **kwargs):
        """
        Wrap the given model with MLRun model features, providing it with MLRun model attributes including its
        parameters and methods.
        :name:
        :param model: The model to wrap.
        :param context: MLRun context to work with. If no context is given it will be retrieved via
                        'mlrun.get_or_create_ctx(None)'
        :param data: The train_test_split X_train, X_test, y_train, y_test.
        :return: The wrapped model.
        """

        # Wrap the fit method:
        def fit_wrapper(fit_method, **kwargs):
            def wrapper(*args, **kwargs):

                train_set = pd.concat([args[0], args[1]], axis=1)
                context.log_dataset('train_set',
                                    df=train_set,
                                    format='csv', index=False,
                                    artifact_path=context.artifact_subpath('data'),
                                    labels={"class": str(model.__class__)},
                                    )

                # Call the original fit method
                fit_method(*args, **kwargs)

                # Original fit method
                setattr(model, "fit", fit_method)
                # Post fit
                if data.get("X_test") is not None:
                    _post_fit(*args, **kwargs)
            return wrapper

        setattr(model, "fit", fit_wrapper(model.fit, **kwargs))

        def _post_fit(*args, **kwargs):
            # Evaluate model results and get the evaluation metrics
            eval_metrics = eval_model_v2(context, data['X_test'], data['y_test'], model)

            # Model Parameters
            model_parameters = {key: str(item) for key, item in model.get_params().items()}
            test_set = pd.concat([data['X_test'], data['y_test']], axis=1)

            # Log test dataset
            context.log_dataset(
                "test_set",
                df=test_set,
                format="parquet",
                index=False,
                labels={"data-type": "held-out"},
                artifact_path=context.artifact_subpath("data"),
            )

            # Log fitted model
            context.set_label("class", str(model.__class__.__name__))
            context.log_model(model_name or "model",
                              db_key=model_name,
                              body=dumps(model),
                              training_set=test_set,
                              artifact_path=context.artifact_subpath("models"),
                              extra_data=eval_metrics,
                              framework=f"{str(model.__module__).split('.')[0]}",
                              algorithm=str(model.__class__.__name__),
                              model_file=f"{str(model.__class__.__name__)}.pkl",
                              metrics=context.results,
                              label_column = data['y_test'].columns.to_list(),
                              )
