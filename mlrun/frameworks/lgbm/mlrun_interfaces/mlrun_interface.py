# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from abc import ABC
from types import ModuleType
from typing import Callable, Optional, Union

import lightgbm as lgb

import mlrun

from ..._common import MLRunInterface
from ..._ml_common import MLArtifactsLibrary, MLTypes
from ..callbacks import Callback, MLRunLoggingCallback
from ..model_handler import LGBMModelHandler
from ..utils import LGBMTypes, LGBMUtils
from .booster_mlrun_interface import LGBMBoosterMLRunInterface


class LGBMMLRunInterface(MLRunInterface, ABC):
    """
    Interface for adding MLRun features for the LightGBM module (Training API).
    """

    # Attributes to be inserted so the MLRun interface will be fully enabled.
    _PROPERTIES = {
        # A list for collecting all the mlrun API callbacks (collected automatically):
        "_mlrun_callbacks": [],  # type: List[Callback]
        # Context to use for logging:
        "_context": None,  # type: mlrun.MLClientCtx
        # Model logging configurations:
        "_log_model": False,
        "_model_handler_kwargs": {},
        "_log_model_kwargs": {},  # Keyword arguments for the model handler's log method.
        # Training logging configurations:
        "_log_training": False,
        "_mlrun_logging_callback_kwargs": {},
    }
    _FUNCTIONS = [
        "configure_logging",
        "_parse_parameters",
        "_parse_callbacks",
        "_pre_train",
        "_post_train",
    ]

    # Attributes to replace so the MLRun interface will be fully enabled.
    _REPLACED_FUNCTIONS = [
        "train",
        # TODO: Wrap `cv` as well.
    ]

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-lightgbm"

    @classmethod
    def add_interface(
        cls,
        obj: Optional[ModuleType] = None,
        restoration: LGBMTypes.MLRunInterfaceRestorationType = None,
    ):
        """
        Enrich the object with this interface properties, methods and functions, so it will have LightGBM MLRun's
        features.

        :param obj:         The object to enrich his interface.
        :param restoration: Restoration information tuple as returned from 'remove_interface' in order to add the
                            interface in a certain state.
        """
        # If the lightgbm module was not provided:
        if obj is None:
            # Set it to the module imported here:
            obj = lgb
            # See if lightgbm was imported outside this file's scope:
            if all(lgb_import not in globals() for lgb_import in ["lightgbm", "lgb"]):
                # Import lightgbm globally:
                globals().update({"lightgbm": lgb, "lgb": lgb})

        # Add the interface to the provided lightgbm module:
        super().add_interface(obj=obj, restoration=restoration)

    @staticmethod
    def mlrun_train(*args, **kwargs):
        """
        MLRun's `lightgbm.train` wrapper. It will parse the parameters and run the training supervised by MLRun.
        """
        # Get the training parameters (must be given) and parse them:
        parameters, _ = MLRunInterface._get_function_argument(
            func=lgb.original_train,
            argument_name="params",
            passed_args=args,
            passed_kwargs=kwargs,
            default_value={},
        )
        lgb._parse_parameters(parameters=parameters)

        # Get the training set (must be given):
        train_set, _ = MLRunInterface._get_function_argument(
            func=lgb.original_train,
            argument_name="train_set",
            passed_args=args,
            passed_kwargs=kwargs,
            default_value=None,
        )
        train_set = (  # Post training LightGBBM is consuming the data, so we keep it in a tuple.
            train_set.data,
            train_set.label,
        )

        # Get the validation sets:
        validation_sets, _ = MLRunInterface._get_function_argument(
            func=lgb.original_train,
            argument_name="valid_sets",
            passed_args=args,
            passed_kwargs=kwargs,
            default_value=[],
        )
        validation_sets_names, _ = MLRunInterface._get_function_argument(
            func=lgb.original_train,
            argument_name="valid_names",
            passed_args=args,
            passed_kwargs=kwargs,
            default_value=[f"valid_{i}" for i in range(len(validation_sets))],
        )
        validation_sets = [
            ((validation_set.data, validation_set.label), name)
            for validation_set, name in zip(validation_sets, validation_sets_names)
        ]

        # Collect the mlrun callbacks from the provided callbacks:
        callbacks, is_given = MLRunInterface._get_function_argument(
            func=lgb.original_train,
            argument_name="callbacks",
            passed_args=args,
            passed_kwargs=kwargs,
            default_value=[],
        )
        lgb._parse_callbacks(callbacks=callbacks)
        if is_given is None:
            kwargs["callbacks"] = callbacks

        # Call the pre-train function:
        lgb._pre_train()

        # Call the original train function:
        booster = lgb.original_train(*args, **kwargs)

        # Call the post train function:
        lgb._post_train(
            booster=booster, train_set=train_set, validation_sets=validation_sets
        )

        return booster

    @staticmethod
    def configure_logging(
        context: mlrun.MLClientCtx = None,
        log_model: bool = True,
        model_handler_kwargs: Optional[dict] = None,
        log_model_kwargs: Optional[dict] = None,
        log_training: bool = True,
        mlrun_logging_callback_kwargs: Optional[dict] = None,
    ):
        """
        Configure the logging of the training API in LightGBM to log the training and model into MLRun. Each `train`
        call will use these configurations to initialize callbacks and log the model at the end of training.

        :param context:                       MLRun context to log to.
        :param log_model:                     Whether to log the model at the end of training. Default: True.
        :param model_handler_kwargs:          Keyword arguments to use for initializing the model handler for the newly
                                              trained model.
        :param log_model_kwargs:              Keyword arguments to use for calling the handler's `log` method.
        :param log_training:                  Whether to log the training metrics and hyperparameters to MLRun.
        :param mlrun_logging_callback_kwargs: Keyword arguments to use for initializing the `MLRunLoggingCallback` for
                                              logging the training results during and post training.

        :raise MLRunInvalidArgumentError: In case the 'model' keyword argument was provided in the
                                          `model_handler_kwargs`.
        """
        # Store the context:
        lgb._context = context

        # Store the given model logging configurations:
        lgb._log_model = log_model
        if model_handler_kwargs is not None:
            if "model" in model_handler_kwargs:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "The 'model' keyword cannot be passed in the model handler keyword arguments as it will be used to "
                    "set the booster that was trained."
                )
            lgb._model_handler_kwargs = model_handler_kwargs
        if log_model_kwargs is not None:
            lgb._log_model_kwargs = log_model_kwargs

        # Store the given training logging configurations:
        lgb._log_training = log_training
        if mlrun_logging_callback_kwargs is not None:
            lgb._mlrun_logging_callback_kwargs = mlrun_logging_callback_kwargs

    @staticmethod
    def _parse_parameters(parameters: dict):
        """
        Parse the parameters passed to the training API functions of LightGBM for enabling MLRun quality of life
        features.

        Notice: The parameters dictionary should be edited and not returned as it is passed by reference.

        :param parameters: The `params` argument that was passed.
        """
        # TODO: Parse `input_model` - if it's set and is a store path, download it locally and set the parameter to the
        #       temp path.
        pass

    @staticmethod
    def _parse_callbacks(callbacks: list[Callable]):
        """
        Parse the callbacks passed to the training API functions of LightGBM for adding logging and enabling the MLRun
        callbacks API.

        Notice: The callbacks list should be edited and not returned as it is passed by reference.

        :param callbacks: The `callbacks` argument that was passed.
        """
        # Check if needed to log training:
        if lgb._log_training:
            # Add only if it was not given already by the user:
            if MLRunLoggingCallback not in [type(callback) for callback in callbacks]:
                callbacks.append(
                    MLRunLoggingCallback(
                        context=lgb._context, **lgb._mlrun_logging_callback_kwargs
                    )
                )

        # Collect all the MLRun API callbacks:
        lgb._mlrun_callbacks.clear()
        for callback in callbacks:
            if isinstance(callback, Callback):
                lgb._mlrun_callbacks.append(callback)

    @staticmethod
    def _pre_train():
        """
        Called pre-training to call the mlrun callbacks `on_train_begin` method.
        """
        for callback in lgb._mlrun_callbacks:
            callback.on_train_begin()

    @staticmethod
    def _post_train(
        booster: lgb.Booster,
        train_set: tuple[MLTypes.DatasetType, Union[MLTypes.DatasetType, None]],
        validation_sets: list[
            tuple[tuple[MLTypes.DatasetType, Union[MLTypes.DatasetType, None]], str]
        ],
    ):
        """
        Called post training to call the mlrun callbacks `on_train_end` method and to log the model.

        :param booster:         The booster to log.
        :param train_set:       The training set that was used to train the given booster.
        :param validation_sets: The validation sets used for training validation.
        """
        # Call the `on_train_end` method of the callbacks while collecting extra data from the mlrun logging callback:
        extra_data = {}
        metrics = {}
        for callback in lgb._mlrun_callbacks:
            callback.on_train_end()
            if isinstance(callback, MLRunLoggingCallback):
                extra_data = {**extra_data, **callback.logger.get_artifacts()}
                metrics = {**metrics, **callback.logger.get_metrics()}

        # Try to produce validation artifacts:  # TODO: Temporary until the Evaluator class will be implemented!
        algorithm_functionality = LGBMUtils.get_algorithm_functionality(
            objective=booster.params.get("objective", "regression")
        )
        plans = [MLArtifactsLibrary.feature_importance()]
        if algorithm_functionality.is_classification():
            plans += [
                MLArtifactsLibrary.calibration_curve(),
                MLArtifactsLibrary.roc_curve(),
                MLArtifactsLibrary.confusion_matrix(),
            ]
        validations_artifacts = {}
        for validation_set, validation_set_name in validation_sets:
            artifacts = {}
            y_pred = booster.predict(validation_set[0])
            plans.append(MLArtifactsLibrary.dataset(name=validation_set_name))
            for plan in plans:
                try:
                    artifacts = {
                        **artifacts,
                        **plan.produce(
                            model=booster,
                            x=validation_set[0],
                            y=validation_set[1],
                            y_pred=y_pred,
                        ),
                    }
                except Exception:
                    pass
            artifacts = {
                f"{validation_set_name}-{key}": value
                for key, value in artifacts.items()
            }
            for artifact in artifacts.values():
                if validation_set_name not in artifact.key:
                    artifact.metadata.key = f"{validation_set_name}-{artifact.key}"
                lgb._context.log_artifact(artifact)
            validations_artifacts = {**validations_artifacts, **artifacts}
            plans.pop(-1)

        # Apply the booster MLRun interface:
        LGBMBoosterMLRunInterface.add_interface(obj=booster)

        # Set the handler to the booster:
        booster.model_handler = LGBMModelHandler(
            model=booster, context=lgb._context, **lgb._model_handler_kwargs
        )

        # Register found extra data and metrics:
        booster.model_handler.set_extra_data(to_add=extra_data)
        booster.model_handler.set_metrics(to_add=metrics)

        # Set the sample set to the training set if None:
        if lgb._log_model_kwargs.get("sample_set", None) is None:
            sample_set, target_columns = LGBMUtils.concatenate_x_y(
                x=train_set[0],
                y=train_set[1],
                target_columns_names=lgb._log_model_kwargs.get("target_columns", None),
            )
            booster.model_handler.set_target_columns(target_columns=target_columns)
            booster.model_handler.set_sample_set(sample_set=sample_set)

        # Check if needed to log the model:
        if lgb._log_model:
            booster.model_handler.register_artifacts(artifacts=validations_artifacts)
            booster.model_handler.log(**lgb._log_model_kwargs)

        lgb._context.commit(completed=False)
