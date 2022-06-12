from abc import ABC
from typing import List
from types import ModuleType

import lightgbm as lgb

import mlrun
from .._common import MLRunInterface
from .._ml_common import MLModelHandler, MLPlan
from ..sklearn import SKLearnMLRunInterface
from .utils import LGBMTypes


class LGBMBoosterMLRunInterface(MLRunInterface, ABC):
    pass


class LGBMModelMLRunInterface(SKLearnMLRunInterface, ABC):
    """
    Interface for adding MLRun features for LightGBM models (SciKit-Learn API models).
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-lgbm"


class LGBMMLRunInterface(MLRunInterface, ABC):
    # Attributes to be inserted so the MLRun interface will be fully enabled.
    _PROPERTIES = {
        "_model_handler": None,  # type: MLModelHandler
        "_logger": 5,  # type: Logger
    }
    _FUNCTIONS = ["_pre_train", "_post_train"]

    # Attributes to replace so the MLRun interface will be fully enabled.
    _REPLACED_FUNCTIONS = ["train"]

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-lgbm"

    @classmethod
    def add_interface(
        cls,
        obj: ModuleType = None,
        restoration: LGBMTypes.MLRunInterfaceRestorationType = None,
    ):
        # If the lightgbm module was not provided:
        if obj is None:
            # Set it to the module imported here:
            obj = lgb
            # See if lightgbm was imported outside this file's scope:
            if all(lgb_import not in globals() for lgb_import in ["lightgbm", "lgb"]):
                # Import lightgbm globally:
                globals().update({"lightgbm": lgb, "lgb": lgb})

        # Add the interface to the provided lightgbm module:
        super(LGBMMLRunInterface, cls).add_interface(
            obj=obj, restoration=restoration
        )

    @staticmethod
    def mlrun_train(*args, **kwargs):
        """
        MLRun's lightgbm.train wrapper. It will setup the optimizer when using horovod. The optimizer must be
        passed in a keyword argument and when using horovod, it must be passed as an Optimizer instance, not a string.

        :raise MLRunInvalidArgumentError: In case the optimizer provided did not follow the instructions above.
        """
        # Get the training parameters:
        if "params" in kwargs:
            parameters = kwargs["params"]
        else:
            parameters = args[0]  # Must be the first argument

        # Call the pre train function:
        parameters = lgb._pre_train(parameters=parameters)

        # Assign parameters:
        if "params" in kwargs:
            kwargs["params"] = parameters
        else:
            # Cast the arguments to a list instead of tuple to override the params argument and cast it back:
            args = list(args)
            args[0] = parameters
            args = tuple(args)

        # Call the original train function:
        model = lgb.original_train(*args, **kwargs)

        # Call the post train function:
        lgb._post_train(model=model)

        return model

    # TODO: Will be removed and replaced in a callback:
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

    # TODO: Will be removed and replaced in a callback:
    def configure_logger(
        self,
        context: mlrun.MLClientCtx = None,
        plans: List[MLPlan] = None,
    ):
        """
        Initialize the MLRun logger for this model using the provided context and artifacts plans, metrics and model
        logging attributes.

        :param context: A MLRun context to log to. By default, uses `mlrun.get_or_create_ctx`
        :param plans:   A list of plans to produce.
        :param metrics: A list of metrics to calculate.
        """
        # Update the MLRun logger:
        if context is None and self._logger.context is None:
            context = mlrun.get_or_create_ctx(
                name=SKLearnMLRunInterface.DEFAULT_CONTEXT_NAME
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

    @staticmethod
    def _pre_train(parameters: dict):
        # TODO: 1. Store the parameters in the model handler. They should be saved as a json artifact and loaded to the
        #          model once it is loaded.
        #       2. If the `input_model` is a store path, load it with a model handler first and pass its booster.
        #       3. Context metrics should be added to the feval (callable and module imports) or metric/s (non module
        #          strings) in parameters.
        pass

    @staticmethod
    def _post_train(model: lgb.Booster):
        # TODO: 1. Generate artifacts.
        #       2. Log the model.
        #       3. Should be moved into a callback!
        pass
