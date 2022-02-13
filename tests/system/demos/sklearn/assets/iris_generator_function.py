import pandas as pd
from sklearn.datasets import load_iris


def iris_generator(context, format="csv"):
    iris = load_iris()
    iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_labels = pd.DataFrame(data=iris.target, columns=["label"])
    iris_dataset = pd.concat([iris_dataset, iris_labels], axis=1)

    context.logger.info(f"saving iris dataframe to {context.artifact_path}")
    context.log_dataset("iris_dataset", df=iris_dataset, format=format, index=False)
