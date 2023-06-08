import pandas as pd

from sklearn import ensemble
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from sklearn.datasets import load_breast_cancer
import warnings
import numpy as np
import mlflow
import logging
import mlrun
from mlrun.frameworks.sklearn import apply_mlrun
import sklearn
from sklearn.linear_model import TweedieRegressor
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

@mlrun.handler()
def train(
    dataset: pd.DataFrame,
    label_column: str = "label",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    model_name: str = "cancer_classifier",
):

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # # mlflow.sklearn.autolog()
    # # Initialize the x & y data
    x = dataset.drop(label_column, axis=1)
    y = dataset[label_column]
    #
    # # Train/Test split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )


    # train, test = train_test_split(dataset)


    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.autolog()
    # Pick an ideal ML model
    model = ensemble.GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth
    )
    # model = TweedieRegressor(power=1, alpha=0.5, link='log')

    # -------------------- The only line you need to add for MLOps -------------------------
    # Wraps the model with MLOps (test set is provided for analysis & accuracy measurements)
    apply_mlrun(model=model, model_name=model_name, x_test=x_test, y_test=y_test)
    # --------------------------------------------------------------------------------------
    # Model registry does not work with file store

    # Train the model
    model.fit(x_train, y_train)

    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # if tracking_url_type_store != "file":
    #     # Register the model
    #     # There are other ways to use the Model Registry, which depends on the use case,
    #     # please refer to the doc for more information:
    #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
    #     mlflow.sklearn.log_model(model, "model", registered_model_name="yoni")
    # else:
    #     mlflow.sklearn.log_model(model, "model")

# if __name__ == '__main__':
#     breast_cancer = load_breast_cancer()
#     breast_cancer_dataset = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
#     breast_cancer_labels = pd.DataFrame(data=breast_cancer.target, columns=["label"])
#     breast_cancer_dataset = pd.concat([breast_cancer_dataset, breast_cancer_labels], axis=1)
#     train(breast_cancer_dataset)