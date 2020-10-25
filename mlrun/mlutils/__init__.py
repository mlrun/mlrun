# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx

from .models import (
    get_class_fit,
    gen_sklearn_model,
    eval_class_model,
    eval_model_v2,
)

# for backwards compatibility
from ..utils.helpers import create_class, create_function

from .plots import (
    gcf_clear,
    feature_importances,
    learning_curves,
    confusion_matrix,
    precision_recall_multi,
    roc_multi,
    roc_bin,
    precision_recall_bin,
    plot_roc,
    plot_importance,
)

from .data import get_sample, get_splits
