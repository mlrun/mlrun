from abc import ABC

from types import ModuleType
import lightgbm as lgbm
import mlrun
from .._common import MLRunInterface, RestorationInformation
from ..sklearn import SKLearnMLRunInterface


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
    _PROPERTIES = {"_logger": 5}
    _FUNCTIONS = ["_pre_train", "_post_train"]

    # Attributes to replace so the MLRun interface will be fully enabled.
    _REPLACED_FUNCTIONS = ["train"]

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-lgbm"

    @classmethod
    def add_interface(
        cls,
        obj: ModuleType = None,
        restoration_information: RestorationInformation = None,
    ):
        # If the lightgbm module was not provided:
        if obj is None:
            # Set it to the module imported here:
            obj = lgbm
            # See if lightgbm was imported outside of this file's scope:
            if "lightgbm" not in globals():
                # Import lightgbm globally:
                globals().update({"lightgbm": lgbm})

        # Add the interface to the provided lightgbm module:
        super(LGBMMLRunInterface, cls).add_interface(
            obj=obj, restoration_information=restoration_information
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
        parameters = lgbm._pre_train(parameters=parameters)

        # Assign parameters:
        if "params" in kwargs:
            kwargs["params"] = parameters
        else:
            # Cast the arguments to a list instead of tuple
            args = list(args)
            args[0] = parameters
            args = tuple(args)

        # Call the original train function:
        model = lgbm.original_train(*args, **kwargs)

        # Call the post train function:
        lgbm._post_train(model=model)

        return model

    @staticmethod
    def _pre_train(parameters: dict):
        # TODO: 1. Store the parameters in the model handler. They should be saved as a json artifact and loaded to the
        #          model once it is loaded.
        #       2. If the `input_model` is a store path, load it with a model handler first and pass its booster.
        #       3. Context metrics should be added to the feval (callable and module imports) or metric/s (non module
        #          strings) in parameters.
        pass

    @staticmethod
    def _post_train(model: lgbm.Booster):
        # TODO: 1. Generate artifacts.
        #       2. Log the model.
        #       3. Should be moved into a callback!
        pass
