from .models import (get_class_fit,
                     create_class,
                     create_function,
                     gen_sklearn_model,
                     eval_class_model,
                     eval_model_v2)

from .plots import (gcf_clear,
                    feature_importances,
                    learning_curves,
                    confusion_matrix,
                    precision_recall_multi,
                    roc_multi,
                    roc_bin,
                    precision_recall_bin,
                    plot_roc,
                    plot_importance)

from .data import (get_sample,
                   get_splits)
