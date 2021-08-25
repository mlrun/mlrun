import mlrun
from mlrun.frameworks.sklearn.mlrun_interface import SklearnMLRunInterface
from mlrun.frameworks.sklearn.model_handler import SklearnModelHandler

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
        context = mlrun.get_or_create_ctx('mlrun_sklearn')
        
    # Add MLRun's interface to the model:
    SklearnMLRunInterface.add_interface(model, context, kwargs)
    return model
