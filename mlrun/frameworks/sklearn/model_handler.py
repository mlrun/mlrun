import os
from typing import Any, Dict, List, Type, Union

import mlrun
from mlrun.artifacts import Artifact
from mlrun.frameworks._common import ModelHandler
import pickle
import joblib


class SklearnModelHandler(ModelHandler):
    """
    Class for handling a sklearn model, enabling loading and saving it during runs.
    """

    class SaveFormats:
        """
        Save formats to pass to the 'SklearnModelHandler'.
        """

        SAVED_MODEL = "SavedModel"
        PICKLE = "pickle"
        JOBLIB = "joblib"
        

        @staticmethod
        def _get_formats() -> List[str]:
            """
            Get a list with all the supported saving formats.
            :return: Saving formats list.
            """
            return [
                value
                for key, value in SklearnModelHandler.SaveFormats.__dict__.items()
                if not key.startswith("_") and isinstance(value, str)
            ]

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        model = None,
        model_path: str = None,
        save_format: str = SaveFormats.PICKLE
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading.
        :param context:        MLRun context to work with.
        :param model:          Model to handle or None in case a loading parameters were supplied.
        :param model_path:     Path to the model directory (SavedModel format) or the model architecture (Json and H5
                               format).
        :param save_format:    The save format to use. Should be passed as a member of the class 'SaveFormats'.
        :raise ValueError: In case the input was incorrect:
                           * Save format is unrecognized.
                           * There was no model or model files supplied.
                           * 'save_traces' parameter was miss-used.
        """
          
        # Load model if path is passed
        if model_path and model is None and save_format == "pickle":
            model = pickle.load(open(model_path, 'rb'))
            
        elif model_path and model is None and save_format == "joblib":
            model = joblib.load(model_path)
            
        elif model is None and model_path is None:
            raise ValueError(f"A model must be loaded, pickle or joblib model file missing.")
        
        # Get model name
        model_name = type(model).__name__
        
            
        super(SklearnModelHandler, self).__init__(
            model=model, context=context
        )

        # Store the configuration:
        self._model_path = model_path
        self._save_format = save_format
        self._model_name = model_name

    def save(
        self, output_path: str = None, *args, **kwargs
    ) -> Union[Dict[str, Artifact], None]:
        """
        Save the handled model at the given output path.
        :param output_path:  The full path to the directory to save the handled model at. If not given, the context
                             stored will be used to save the model in the defaulted location.
        :return The saved model artifacts dictionary if context is available and None otherwise.
        :raise RuntimeError: In case there is no model initialized in this handler.
        :raise ValueError:   If an output path was not given, yet a context was not provided in initialization.
        """
        super(SklearnModelHandler, self).save(output_path=output_path)
        
        # Set the output path:
        if output_path is None:
            output_path = os.path.join(self._context.artifact_path, self._model_name)
            
        # Setup the returning model artifacts list:
        artifacts = {}  # type: Dict[str, Artifact]


        if self._save_format == self.SaveFormats.PICKLE:
            pickle.dump(model, open(output_path, 'wb'))
            
        elif self._save_format == self.SaveFormats.JOBLIB:
            joblib.dump(model, output_path)
        
        
        # Update the paths and log artifacts if context is available:
        if self._context and self._model_path:
            artifacts["model_file"] = self._context.log_artifact(
                self._model_path,
                local_path=self._model_path,
                artifact_path=output_path,
                db_key=False,
            )
                
        return artifacts if self._context else None

    def load(self, uid: str = None, *args, **kwargs):
        """
        Load the specified model in this handler. Additional parameters for the class initializer can be passed via the
        args list and kwargs dictionary.
        """
        super(SklearnModelHandler, self).load(uid=uid)


    def log(
        self,
        labels: Dict[str, Union[str, int, float]],
        parameters: Dict[str, Union[str, int, float]],
        extra_data: Dict[str, Any],
        artifacts: Dict[str, Artifact],
    ):
        """
        Log the model held by this handler into the MLRun context provided.
        :param labels:     Labels to log the model with.
        :param parameters: Parameters to log with the model.
        :param extra_data: Extra data to log with the model.
        :param artifacts:  Artifacts to log the model with.
        :raise RuntimeError: In case there is no model in this handler.
        :raise ValueError:   In case a context is missing.
        """
        super(SklearnModelHandler, self).log(
            labels=labels,
            parameters=parameters,
            extra_data=extra_data,
            artifacts=artifacts,
        )

        # Save the model:
        model_artifacts = self.save(update_paths=True)

        # Log the model:
        self._context.log_model(
            self._model.name,
            model_file=self._model_path,
            framework="tensorflow.keras",
            labels={"save-format": self._save_format, **labels},
            parameters=parameters,
            metrics=self._context.results,
            extra_data={**model_artifacts, **artifacts, **extra_data},
        )
        # Log model
        self._context.log_model("model",
                          body=dumps(model),
                          parameters = model_parameters,
                          artifact_path=self._context.artifact_subpath("models"),
                          extra_data = eval_metrics,
                          model_file="model.pkl",
                          metrics=self._context.results,
                          labels={"class": str(model.__class__)})
            
