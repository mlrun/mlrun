"""
Description:
__init__ function of sklearn-autologger. Will be extended and contain multiple Sklearn-specific functions.
"""

import mlrun
from mlrun.frameworks.mlbase.mlrun_interface import MLBaseMLRunInterface
from mlrun.frameworks._common.pkl_model_server import PklModelServer

# Temporary placeholder, SklearnModelServer may deviate from PklModelServer in upcoming versions.
SklearnModelServer = PklModelServer
        
def apply_mlrun(
        model,
        context: mlrun.MLClientCtx = None,
        **kwargs):
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.
    
    examples::          model = LogisticRegression()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                        model = apply_mlrun_xgb(model, context, X_test=X_test, y_test=y_test)
                        model.fit(X_train, y_train)
   
    :param model:       The model to wrap.
    
    :param context:     MLRun context to work with. If no context is given it will be retrieved via
                        'mlrun.get_or_create_ctx(None)'
                        
    :return: The model with MLRun's interface.
    """
    if context is None:
        context = mlrun.get_or_create_ctx('mlrun_sklearn')
         
    # Add MLRun's interface to the model:
    MLBaseMLRunInterface.add_interface(model, context, kwargs)
    return model


       """update or add a function object to the project

        function can be provided as an object (func) or a .py/.ipynb/.yaml url

        supported url prefixes::

            object (s3://, v3io://, ..)
            MLRun DB e.g. db://project/func:ver
            functions hub/market: e.g. hub://sklearn_classifier:master

        examples::

            proj.set_function(func_object)
            proj.set_function('./src/mycode.py', 'ingest',
                              image='myrepo/ing:latest', with_repo=True)
            proj.set_function('http://.../mynb.ipynb', 'train')
            proj.set_function('./func.yaml')
            proj.set_function('hub://get_toy_data', 'getdata')

        :param func:      function object or spec/code url
        :param name:      name of the function (under the project)
        :param kind:      runtime kind e.g. job, nuclio, spark, dask, mpijob
                          default: job
        :param image:     docker image to be used, can also be specified in
                          the function object/yaml
        :param with_repo: add (clone) the current repo to the build source

        :returns: project object
        """
