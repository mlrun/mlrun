import pandas as pd
from sklearn.datasets import load_breast_cancer


def breast_cancer_generator(context, format="csv"):
    """a function which generates the breast cancer dataset"""
    breast_cancer = load_breast_cancer()
    breast_cancer_dataset = pd.DataFrame(
        data=breast_cancer.data, columns=breast_cancer.feature_names
    )
    breast_cancer_labels = pd.DataFrame(data=breast_cancer.target, columns=["label"])
    breast_cancer_dataset = pd.concat(
        [breast_cancer_dataset, breast_cancer_labels], axis=1
    )

    context.logger.info("saving breast cancer dataframe")
    context.log_result("label_column", "label")
    context.log_dataset("dataset", df=breast_cancer_dataset, format=format, index=False)
