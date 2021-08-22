class SklearnMLRunInterface(MLRunInterface):
    """
    MLRun model is for enabling additional features supported by MLRun in keras. With MLRun model one can apply horovod
    and use auto logging with ease.
    """
    
    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-sklearn"
    
    @classmethod
    def add_interface(cls, model, context, *args, **kwargs):
        """
        Wrap the given model with MLRun model features, providing it with MLRun model attributes including its
        parameters and methods.
        :param model: The model to wrap.
        :return: The wrapped model.
        """

        # Wrap the fit method:
        def fit_wrapper(fit_method):
            def wrapper(*args, **kwargs):
                print('X_train',args[0])
                print('X_test',args[1])
            
                context.log_dataset('train_set', 
                                    df=pd.concat([args[0], args[1]], axis=1),
                                    format='csv', index=False, 
                                    artifact_path=context.artifact_subpath('data'))
                
                # Call the original fit method
                fit_method(*args, **kwargs)
                
                # Original fit method
                setattr(model, "fit", fit_method)
                
                # Post fit
                post_fit(*args, **kwargs)
            return wrapper
        setattr(model, "fit", fit_wrapper(model.fit))
        
        def post_fit(*args, **kwargs):
            
            # Evaluate model results and get the evaluation metrics
            eval_metrics = eval_model_v2(context, X_test, y_test, model)
            

            # Model Parameters
            model_parameters = {key: str(item) for key,item in model.get_params().items()}      
                        
            # Log model
            context.log_model("model",
                              body=dumps(model),
                              parameters = model_parameters,
                              artifact_path=context.artifact_subpath("models"),
                              extra_data = eval_metrics,
                              model_file="model.pkl",
                              metrics=context.results,
                              labels={"class": str(model.__class__)})