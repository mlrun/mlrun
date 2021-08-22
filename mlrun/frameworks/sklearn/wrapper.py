from mlrun.frameworks._common import MLRunInterface

from typing import Union
from sklearn.model_selection import train_test_split

from cloudpickle import dumps
from mlrun.mlutils.plots import eval_model_v2
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import mlrun

Model = Union[LogisticRegression, str]

class SklearnMLRunInterface(MLRunInterface):
    """
    MLRun model is for enabling additional features supported by MLRun in keras. With MLRun model one can apply horovod
    and use auto logging with ease.
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-sklearn"

    @classmethod
    def add_interface(cls, model: Model, context, *args, **kwargs):
        """
        Wrap the given model with MLRun model features, providing it with MLRun model attributes including its
        parameters and methods.
        :param model: The model to wrap.
        :return: The wrapped model.
        """

        # Wrap the fit method:
        def fit_wrapper(fit_method):
            def wrapper(*args, **kwargs):
                context.log_dataset('train_set',
                                    df=pd.concat([X_train, y_train], axis=1),
                                    format='csv', index=False,
                                    artifact_path=context.artifact_subpath('data'))

                context.log_dataset('test_set',
                                    df=pd.concat([X_test, y_test], axis=1),
                                    format='csv', index=False,
                                    labels={"data-type": "held-out"},
                                    artifact_path=context.artifact_subpath('data'))

                # Call the original fit method
                fit_method(*args, **kwargs)

                # Original fit method
                setattr(model, "fit", fit_method)

                # Post fit
                post_fit(*args, **kwargs)

            return wrapper
        setattr(model, "fit", fit_wrapper(model.fit))

        def post_fit(*args, **kwargs):
            # Evaluate model results and get the evaluation metrics
            eval_metrics = eval_model_v2(context, X_test, y_test, model)

            # Model Parameters
            model_parameters = {key: str(item) for key, item in model.get_params().items()}

            print(model_parameters)

            # Log model
            context.log_model("model",
                              body=dumps(model),
                              parameters=model_parameters,
                              artifact_path=context.artifact_subpath("models"),
                              extra_data=eval_metrics,
                              model_file="model.pkl",
                              metrics=context.results,
                              labels={"class": str(model.__class__)})


def apply_mlrun(model, context=None):
    if context is None:
        context = mlrun.get_or_create_ctx(SklearnMLRunInterface.DEFAULT_CONTEXT_NAME)

    # Add MLRun's interface to the model:
    SklearnMLRunInterface.add_interface(model, context)
    return model

def classification():
    classification_models = [
    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB()]

    # Load Iris Data
    iris = load_iris()

    X = pd.DataFrame(data= iris.data, columns= iris.feature_names)
    y = pd.DataFrame(data= iris.target, columns= ['species'])

    # Basic scikit-learn Iris data-set SVM model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    for m in classification_models:
        model = apply_mlrun(m)
        model.fit(X_train, y_train.values.reshape(-1,))
        pred = model.predict(X_test)
        print(model, pred)

classification()