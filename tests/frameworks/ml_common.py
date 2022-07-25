from typing import Tuple, Union

from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.model_selection import train_test_split

from mlrun.frameworks._ml_common import AlgorithmFunctionality, MLTypes

N_TARGETS = 5
N_CLASSES = 5
N_SAMPLES = 200


def get_dataset(
    algorithm_functionality: AlgorithmFunctionality,
    for_training: bool,
    n_targets: int = N_TARGETS,
    n_classes: int = N_CLASSES,
    n_samples: int = N_SAMPLES,
) -> Union[
    Tuple[MLTypes.DatasetType, MLTypes.DatasetType],
    Tuple[
        MLTypes.DatasetType,
        MLTypes.DatasetType,
        MLTypes.DatasetType,
        MLTypes.DatasetType,
    ],
]:
    if algorithm_functionality.is_regression():
        if algorithm_functionality.is_single_output():
            n_targets = 1
        x, y = make_regression(n_samples=n_samples, n_targets=n_targets)
        stratify = None
    else:
        if algorithm_functionality.is_binary_classification():
            n_classes = 2
        if algorithm_functionality.is_single_output():
            x, y = make_classification(
                n_samples=n_samples, n_classes=n_classes, n_informative=n_classes
            )
            stratify = y
        else:
            x, y = make_multilabel_classification(
                n_samples=n_samples, n_classes=n_classes
            )
            stratify = None

    if not for_training:
        return x, y
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=stratify
    )
    return x_train, x_test, y_train, y_test
