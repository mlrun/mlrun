import sklearn
import mlrun
from mlrun.frameworks.sklearn.mlrun_interface import SklearnMLRunInterface
from mlrun.frameworks.sklearn.model_handler import SklearnModelHandler

def apply_mlrun(
    model,
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    ):
    
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
    
    
    if context is None:
        context = mlrun.get_or_create_ctx(SklearnMLRunInterface.DEFAULT_CONTEXT_NAME)
        
    # Add MLRun's interface to the model:
    SklearnMLRunInterface.add_interface(model=model, context=context)
    
    # Add auto-logging if needed:
    if auto_log:
        model.auto_log(context=context)
        
    return model