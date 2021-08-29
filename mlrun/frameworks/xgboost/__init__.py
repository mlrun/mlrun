import mlrun
from mlrun.frameworks.xgboost.mlrun_interface import XGBMLRunInterface
from mlrun.frameworks._common.pkl_model_server import PklModelServer

def apply_mlrun(
        model,
        context: mlrun.MLClientCtx = None,
        **kwargs):
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.
    :param model:       The model to wrap.
    :param context:     MLRun context to work with. If no context is given it will be retrieved via
                        'mlrun.get_or_create_ctx(None)'
    :return: The model with MLRun's interface.
    """
    if context is None:
        context = mlrun.get_or_create_ctx('mlrun_xgb')

    XGBModelServer = PklModelServer

    # Add MLRun's interface to the model:
    XGBMLRunInterface.add_interface(model, context, kwargs)
    return model
