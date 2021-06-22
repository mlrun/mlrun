# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx
from tensorflow import keras

import mlrun
import mlrun.frameworks.keras.callbacks
from mlrun.frameworks.keras.mlrun_interface import KerasMLRunInterface
from mlrun.frameworks.keras.model_handler import KerasModelHandler


def apply_mlrun(
    model: keras.Model,
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    use_horovod: bool = None,
) -> keras.Model:
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.

    :param model:       The model to wrap.
    :param context:     MLRun context to work with. If no context is given it will be retrieved via
                        'mlrun.get_or_create_ctx(None)'
    :param auto_log:    Whether or not to apply MLRun's auto logging on the model. Defaulted to True.
    :param use_horovod: Whether or not to use horovod - a distributed training framework. Defaulted to None, meaning it
                        will be read from context if available and if not - False.

    :return: The model with MLRun's interface.
    """
    # Get parameters defaults:
    # # Context:
    if context is None:
        context = mlrun.get_or_create_ctx(KerasMLRunInterface.DEFAULT_CONTEXT_NAME)
    # # Use horovod:
    if use_horovod is None:
        use_horovod = (
            context.labels.get("kind", "") == "mpijob" if context is not None else False
        )

    # Add MLRun's interface to the model:
    KerasMLRunInterface.add_interface(model=model)

    # Initialize horovod if needed:
    if use_horovod is True:
        model.use_horovod()

    # Add auto-logging if needed:
    if auto_log:
        model.auto_log(context=context)

    return model
