import pandas as pd
from mlrun.frameworks._common.plots import eval_model_v2
from mlrun.frameworks._common import MLRunInterface
from cloudpickle import dumps

class MLBaseMLRunInterface(MLRunInterface):
    """
    Base run interface for ML frameworks (sklearn, xgb, ..)
    Wraps the original .fit() method of the passed model enabling auto-logging.
    """

    @classmethod
    def add_interface(cls, model, context, data, *args, **kwargs):
        """
        Wrap the given model with MLRun model features, providing it with MLRun model attributes including its
        parameters and methods.
        :param model: The model to wrap.
        :param context: MLRun context to work with. If no context is given it will be retrieved via
                        'mlrun.get_or_create_ctx(None)'
        :param data: The train_test_split X_train, X_test, y_train, y_test.
        :return: The wrapped model.
        """

        # Wrap the fit method:
        def fit_wrapper(fit_method, **kwargs):
            def wrapper(*args, **kwargs):
                context.log_dataset('train_set',
                                    df=pd.concat([data['X_train'], data['y_train']], axis=1),
                                    format='csv', index=False,
                                    artifact_path=context.artifact_subpath('data'))

                # Call the original fit method
                fit_method(*args, **kwargs)

                # Original fit method
                setattr(model, "fit", fit_method)
                # Post fit
                if data.get("X_test") is not None:
                    post_fit(*args, **kwargs)
            return wrapper

        setattr(model, "fit", fit_wrapper(model.fit, **kwargs))

        def post_fit(*args, **kwargs):
            # Evaluate model results and get the evaluation metrics
            eval_metrics = eval_model_v2(context, data['X_test'], data['y_test'], model)

            # Model Parameters
            model_parameters = {key: str(item) for key, item in model.get_params().items()}

            # Log model
            context.log_model("model",
                              body=dumps(model),
                              parameters=model_parameters,
                              artifact_path=context.artifact_subpath("models"),
                              extra_data=eval_metrics,
                              model_file=f"{str(type(model).__name__)}.pkl",
                              metrics=context.results,
                              labels={"class": str(model.__class__)})
