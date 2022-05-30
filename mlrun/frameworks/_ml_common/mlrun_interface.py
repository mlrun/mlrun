from abc import ABC
from typing import List

import mlrun

from .._common import MLRunInterface, ModelType, RestorationInformation
from .logger import Logger, LoggerMode
from .metrics_library import Metric
from .model_handler import MLModelHandler
from .plan import MLPlan, MLPlanStages
from .utils import DatasetType, concatenate_x_y


class MLMLRunInterface(MLRunInterface, ABC):
    """
    Interface for adding MLRun features for machine learning common API.
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-ml"

    # Attributes to be inserted so the MLRun interface will be fully enabled.
    _PROPERTIES = {
        # A model handler instance with the model for logging / updating the model (if not provided the model won't be
        # logged / updated at the end of training / testing):
        "_model_handler": None,  # > type: MLModelHandler
        # The logger that is logging this model's training / evaluation:
        "_logger": None,  # > type: Logger
        # The test set (For validation post training or evaluation post prediction):
        "_x_test": None,  # > type: DatasetType
        "_y_test": None,  # > type: DatasetType
    }
    _METHODS = [
        "set_model_handler",
        "configure_logger",
        "_pre_fit",
        "_post_fit",
        "_pre_predict",
        "_post_predict",
    ]

    # Attributes to replace so the MLRun interface will be fully enabled.
    _REPLACED_METHODS = ["fit", "predict", "predict_proba"]

    @classmethod
    def add_interface(
        cls,
        obj: ModelType,
        restoration_information: RestorationInformation = None,
    ):
        """
        Enrich the object with this interface properties, methods and functions so it will have this framework MLRun's
        features.

        :param obj:                     The model object to enrich his interface.
        :param restoration_information: Restoration information tuple as returned from 'remove_interface' in order to
                                        add the interface in a certain state.
        """
        # Check if the given model has the 'predict_proba' method to replace:
        if not hasattr(obj, "predict_proba"):
            cls._REPLACED_METHODS.remove("predict_proba")

        # Add the interface to the model:
        super(MLMLRunInterface, cls).add_interface(
            obj=obj, restoration_information=restoration_information
        )

        # Restore the '_REPLACED_METHODS' list for next models:
        if "predict_proba" not in cls._REPLACED_METHODS:
            cls._REPLACED_METHODS.append("predict_proba")

        # Setup a default logger:
        if obj._logger is None:
            obj._logger = Logger()

    @classmethod
    def mlrun_fit(cls):
        """
        MLRun's common ML API fit wrapper. It will run a validation post training if provided the required attributes.

        Unsupervised learning algorithms won't be using 'y' in their 'fit' method, but for consistency reasons (as
        written in SciKit-Learn's API documentation): the signature of their 'fit' method will still have 'y'. For that
        we do not need another wrapper that support unsupervised learning models.
        """

        def wrapper(
            self: ModelType, X: DatasetType, y: DatasetType = None, *args, **kwargs
        ):
            # Restore the prediction methods as fit will use them:
            cls._restore_attribute(obj=self, attribute_name="predict")
            if hasattr(self, "predict_proba"):
                cls._restore_attribute(obj=self, attribute_name="predict_proba")

            # Call the pre fit method:
            self._pre_fit(x=X, y=y)

            # Call the original fit method:
            result = self.original_fit(X, y, *args, **kwargs)

            # Call the post fit method:
            self._post_fit(x=X, y=y)

            # Replace the prediction methods again:
            cls._replace_function(obj=self, function_name="predict")
            if hasattr(self, "predict_proba"):
                cls._replace_function(obj=self, function_name="predict_proba")
            return result

        return wrapper

    def mlrun_predict(self, X: DatasetType, *args, **kwargs):
        """
        MLRun's wrapper for the common ML API predict method.
        """
        self._pre_predict(x=X, y=self._y_test)

        y_pred = self.original_predict(X, *args, **kwargs)

        self._post_predict(x=X, y=self._y_test, y_pred=y_pred, is_predict_proba=False)

        return y_pred

    def mlrun_predict_proba(self, X: DatasetType, *args, **kwargs):
        """
        MLRun's wrapper for the common ML API predict_proba method.
        """
        self._pre_predict(x=X, y=self._y_test)

        y_pred = self.original_predict_proba(X, *args, **kwargs)

        self._post_predict(x=X, y=self._y_test, y_pred=y_pred, is_predict_proba=True)

        return y_pred

    def set_model_handler(self, model_handler: MLModelHandler):
        """
        Set this model's MLRun handler for logging the model as a model artifact post training (post calling 'fit') or
        update the existing model artifact post testing (calling 'predict' / 'predict_proba'). If the logger's context
        is None, it will set its context to the handler's context.

        :param model_handler: The ML model handler with a loaded model.
        """
        # Store the given model handler:
        self._model_handler = model_handler

        # Update the logger's context to the handler's context if its None:
        if self._logger.context is None:
            self._logger.set_context(context=model_handler.context)

    def configure_logger(
        self,
        context: mlrun.MLClientCtx = None,
        plans: List[MLPlan] = None,
        metrics: List[Metric] = None,
        x_test: DatasetType = None,
        y_test: DatasetType = None,
    ):
        """
        Initialize the MLRun logger for this model using the provided context and artifacts plans, metrics and model
        logging attributes.

        :param context: A MLRun context to log to. By default, uses `mlrun.get_or_create_ctx`
        :param plans:   A list of plans to produce.
        :param metrics: A list of metrics to calculate.
        :param x_test:  The testing data for producing and calculating artifacts and metrics post training or post
                        prediction. Without this, training validation will not be performed.
        :param y_test:  The test data for producing and calculating artifacts and metrics post training (calling 'fit')
                        or post testing (calling 'predict' / 'predict_proba').
        """
        # Update the MLRun logger:
        if context is None and self._logger.context is None:
            context = mlrun.get_or_create_ctx(
                name=MLMLRunInterface.DEFAULT_CONTEXT_NAME
            )
        if context is not None:
            self._logger.set_context(context=context)
        self._logger.set_plans(plans=plans)
        self._logger.set_metrics(metrics=metrics)

        # Validate that if the prediction probabilities are required, this model has the 'predict_proba' method:
        if self._logger.is_probabilities_required() and not hasattr(
            self, "predict_proba"
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Some of the metrics and or artifacts required to be calculated and produced require prediction "
                f"probabilities yet this model: '{type(self)}' do not has the 'predict_proba' method."
            )

        # Store the given dataset:
        self._x_test = x_test
        self._y_test = y_test

    def _pre_fit(self, x: DatasetType, y: DatasetType = None):
        """
        Method for creating the artifacts before the fit method.

        :param x: The input dataset to the fit method ('x_train').
        :param y: The input dataset to the fit method ('y_train').
        """
        self._logger.log_stage(stage=MLPlanStages.PRE_FIT, model=self, x=x, y=y)

    def _post_fit(self, x: DatasetType, y: DatasetType = None):
        """
        Method for creating the artifacts after the fit method. If a validation set is available, the method will start
        a validation process calling predict - creating and calculating validation artifacts and metrics.

        :param x: The input dataset to the fit method ('x_train').
        :param y: The input dataset to the fit method ('y_train').
        """
        # The model is done training, log all artifacts post fit:
        self._logger.log_stage(stage=MLPlanStages.POST_FIT, model=self, x=x, y=y)

        # If there is a validation set, run validation:
        if self._x_test is not None:
            self._logger.log_stage(
                stage=MLPlanStages.PRE_PREDICT,
                model=self,
                x=self._x_test,
                y=self._y_test,
            )
            y_pred = self.predict(self._x_test)
            self._post_predict(
                x=self._x_test,
                y=self._y_test,
                y_pred=y_pred,
                is_predict_proba=False,
            )

        # Log the model with the given attributes:
        if self._model_handler is not None:
            # Set the sample set to the training set if None:
            if self._model_handler.sample_set is None:
                sample_set, y_columns = concatenate_x_y(
                    x=x, y=y, y_columns=self._model_handler.y_columns
                )
                self._model_handler.set_y_columns(y_columns=y_columns)
                self._model_handler.set_sample_set(sample_set=sample_set)
            # Log the model:
            self._logger.log_run(model_handler=self._model_handler)

    def _pre_predict(self, x: DatasetType, y: DatasetType):
        """
        Method for creating the artifacts before the predict method.

        :param x: The input dataset to the predict / predict_proba method ('x_test').
        :param y: The input dataset to the predict / predict_proba method ('y_test').
        """
        # This function is only called for evaluation, then set the logger's mode:
        self._logger.set_mode(mode=LoggerMode.EVALUATION)

        # Produce and log all the artifacts pre prediction:
        self._logger.log_stage(stage=MLPlanStages.PRE_PREDICT, model=self, x=x, y=y)

    def _post_predict(
        self,
        x: DatasetType,
        y: DatasetType,
        y_pred: DatasetType,
        is_predict_proba: bool,
    ):
        """
        Method for creating and calculating the artifacts and metrics after the predict method. This method can be
        called after a user call to predict as part of testing or as part of validation after training (calling fit).

        :param x:                The input dataset to the predict / predict_proba method ('x_test' / 'x_validation').
        :param y:                The input dataset to the predict / predict_proba method ('y_test' / 'y_validation').
        :param y_pred:           The prediction outcome.
        :param is_predict_proba: Whether the prediction returned from predict or predict_proba.
        """
        # Produce and log all the artifacts post prediction:
        self._logger.log_stage(
            stage=MLPlanStages.POST_PREDICT,
            model=self,
            x=x,
            y=y,
            y_pred=y_pred,
            is_probabilities=is_predict_proba,
        )

        # Calculate and log the metrics results:
        self._logger.log_results(
            y_true=y, y_pred=y_pred, is_probabilities=is_predict_proba
        )

        # If some metrics and / or plans require probabilities, run 'predict_proba':
        if not is_predict_proba and self._logger.is_probabilities_required():
            y_pred_proba = self.predict_proba(x)
            self._logger.log_stage(
                stage=MLPlanStages.POST_PREDICT,
                is_probabilities=True,
                model=self,
                x=x,
                y=y,
                y_pred=y_pred_proba,
            )
            self._logger.log_results(
                y_true=y, y_pred=y_pred_proba, is_probabilities=True
            )

        # If its part of validation post training, return:
        if self._logger.mode == LoggerMode.TRAINING:
            return

        # Update the model with the testing artifacts and results:
        if self._model_handler is not None:
            self._logger.log_run(model_handler=self._model_handler)
