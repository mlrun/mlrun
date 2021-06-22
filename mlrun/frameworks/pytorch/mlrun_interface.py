import importlib
import sys
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.multiprocessing as mp
from tabulate import tabulate
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import mlrun
from mlrun.frameworks.pytorch.callbacks import (
    Callback,
    HyperparametersKeys,
    MetricFunctionType,
    MetricValueType,
    MLRunLoggingCallback,
    TensorboardLoggingCallback,
)
from mlrun.frameworks.pytorch.callbacks_handler import CallbacksHandler


class PyTorchMLRunInterface:
    """
    An interface for enabling convenient MLRun features for the PyTorch framework, including training, evaluating and
    automatic logging.
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-pytorch"

    def __init__(self, model: Module, context: mlrun.MLClientCtx = None):
        """
        Initialize an interface for running training and evaluation on the given parameters.

        :param model:   The model to train / evaluate.
        :param context: MLRun context to use. If None, the context will be taken from 'mlrun.get_or_create_ctx()'.
        """
        # Set the context:
        if context is None:
            context = mlrun.get_or_create_ctx(self.DEFAULT_CONTEXT_NAME)

        # Store the model:
        self._model = model
        self._context = context

        # Prepare methods parameters:
        self._training_set = None  # type: DataLoader
        self._loss_function = None  # type: Module
        self._optimizer = None  # type: Optimizer
        self._validation_set = None  # type: DataLoader
        self._metric_functions = None  # type: List[MetricFunctionType]
        self._scheduler = None
        self._scheduler_step_frequency = None  # type: int
        self._epochs = None  # type: int
        self._training_iterations = None  # type: int
        self._validation_iterations = None  # type: int
        self._callbacks = []  # type: List[Callback]
        self._use_cuda = None  # type: bool
        self._use_horovod = None  # type: bool

        # Prepare inner attributes:
        self._model_in_cuda = False
        self._hvd = None
        self._training_sampler = None  # type: DistributedSampler
        self._validation_sampler = None  # type: DistributedSampler
        self._callbacks_handler = None  # type: CallbacksHandler

    def train(
        self,
        training_set: DataLoader,
        loss_function: Module,
        optimizer: Optimizer,
        validation_set: DataLoader = None,
        metric_functions: List[MetricFunctionType] = None,
        scheduler=None,
        scheduler_step_frequency: Union[int, float, str] = "epoch",
        epochs: int = 1,
        training_iterations: int = None,
        validation_iterations: int = None,
        callbacks: List[Callback] = None,
        use_cuda: bool = True,
        use_horovod: bool = None,
    ):
        """
        Initiate a training process on this interface configuration.

        :param training_set:             A data loader for the training process.
        :param loss_function:            The loss function to use during training.
        :param optimizer:                The optimizer to use during the training.
        :param validation_set:           A data loader for the validation process.
        :param metric_functions:         The metrics to use on training and validation.
        :param scheduler:                Scheduler to use on the optimizer at the end of each epoch. The scheduler must
                                         have a 'step' method with no input.
        :param scheduler_step_frequency: The frequency in which to step the given scheduler. Can be equal to one of the
                                         strings 'epoch' (for at the end of every epoch) and 'batch' (for at the end of
                                         every batch), or an integer that specify per how many iterations to step or a
                                         float percentage (0.0 < x < 1.0) for per x / iterations to step. Defaulted to
                                         'epoch'.
        :param epochs:                   Amount of epochs to perform. Defaulted to a single epoch.
        :param training_iterations:      Amount of iterations (batches) to perform on each epoch's training. If 'None'
                                         the entire training set will be used.
        :param validation_iterations:    Amount of iterations (batches) to perform on each epoch's validation. If 'None'
                                         the entire validation set will be used.
        :param callbacks:                The callbacks to use on this run.
        :param use_cuda:                 Whether or not to use cuda. Only relevant if cuda is available. Defaulted to
                                         True.
        :param use_horovod:              Whether or not to use horovod - a distributed training framework. Defaulted to
                                         None, meaning it will be read from context if available and if not - False.
        """
        # Load the input:
        self._parse_and_store(
            training_set=training_set,
            loss_function=loss_function,
            optimizer=optimizer,
            validation_set=validation_set,
            metric_functions=metric_functions,
            scheduler=scheduler,
            scheduler_step_frequency=scheduler_step_frequency,
            epochs=epochs,
            training_iterations=training_iterations,
            validation_iterations=validation_iterations,
            callbacks=callbacks,
            use_cuda=use_cuda,
            use_horovod=use_horovod,
        )

        # Setup the inner attributes (initializing horovod and creating the callbacks handler):
        self._setup()

        # Beginning of run callbacks:
        self._callbacks_handler.on_run_begin()

        # Start the epochs:
        for epoch in range(self._epochs):
            # Beginning of a epoch callbacks:
            self._callbacks_handler.on_epoch_begin(epoch=epoch)
            print(
                "Epoch {}/{}:".format(
                    str(epoch + 1).rjust(len(str(self._epochs))), self._epochs
                )
            )

            # Train:
            self._callbacks_handler.on_train_begin()
            self._train()
            if not self._callbacks_handler.on_train_end():
                break

            # Validate:
            if self._validation_set is not None:
                self._callbacks_handler.on_validation_begin()
                loss_value, metric_values = self._validate()
                # If horovod is used, wait for all ranks to calculate the loss and metrics averages:
                if self._use_horovod:
                    loss_value = self._metric_average(
                        rank_value=loss_value,
                        name="average_{}".format(
                            self._get_metric_name(metric=self._loss_function)
                        ),
                    )
                    metric_values = [
                        self._metric_average(
                            rank_value=metric_value,
                            name="average_{}".format(
                                self._get_metric_name(metric=metric_function)
                            ),
                        )
                        for metric_value, metric_function in zip(
                            metric_values, self._metric_functions
                        )
                    ]
                self._print_results(loss_value=loss_value, metric_values=metric_values)
                if not self._callbacks_handler.on_validation_end(
                    loss_value=loss_value, metric_values=metric_values
                ):
                    break

            # End of a epoch callbacks:
            if not self._callbacks_handler.on_epoch_end(epoch=epoch):
                break
            print()

        # End of run callbacks:
        self._callbacks_handler.on_run_end()

        # Clear the interface:
        self._clear()

    def evaluate(
        self,
        dataset: DataLoader,
        loss_function: Module = None,
        metric_functions: List[MetricFunctionType] = None,
        iterations: int = None,
        callbacks: List[Callback] = None,
        use_cuda: bool = True,
        use_horovod: bool = None,
    ) -> List[MetricValueType]:
        """
        Initiate an evaluation process on this interface configuration.

        :param dataset:          A data loader for the validation process.
        :param loss_function:    The loss function to use during training.
        :param metric_functions: The metrics to use on training and validation.
        :param iterations:       Amount of iterations (batches) to perform on the dataset. If 'None' the entire dataset
                                 will be used.
        :param callbacks:        The callbacks to use on this run.
        :param use_cuda:         Whether or not to use cuda. Only relevant if cuda is available. Defaulted to True.
        :param use_horovod:      Whether or not to use horovod - a distributed training framework. Defaulted to None,
                                 meaning it will be read from context if available and if not - False.

        :return: The evaluation loss and metrics results in a list.
        """
        # Load the input:
        self._parse_and_store(
            validation_set=dataset,
            loss_function=loss_function,
            metric_functions=metric_functions,
            validation_iterations=iterations,
            callbacks=callbacks,
            use_cuda=use_cuda,
            use_horovod=use_horovod,
        )

        # Setup the inner attributes (initializing horovod and creating the callbacks handler):
        self._setup()

        # Beginning of run callbacks:
        self._callbacks_handler.on_run_begin()

        # Evaluate:
        self._callbacks_handler.on_validation_begin()
        loss_value, metric_values = self._validate(is_evaluation=True)

        # If horovod is used, wait for all ranks to calculate the loss and metrics averages:
        if self._use_horovod:
            loss_value = self._metric_average(
                rank_value=loss_value,
                name="average_{}".format(
                    self._get_metric_name(metric=self._loss_function)
                ),
            )
            metric_values = [
                self._metric_average(
                    rank_value=metric_value,
                    name="average_{}".format(
                        self._get_metric_name(metric=metric_function)
                    ),
                )
                for metric_value, metric_function in zip(
                    metric_values, self._metric_functions
                )
            ]

        # End the validation:
        self._print_results(loss_value=loss_value, metric_values=metric_values)
        self._callbacks_handler.on_validation_end(
            loss_value=loss_value, metric_values=metric_values
        )
        print()

        # End of run callbacks:
        self._callbacks_handler.on_run_end()

        # Clear the interface:
        self._clear()

        return [loss_value] + metric_values

    def add_auto_logging_callbacks(
        self,
        custom_objects: Dict[Union[str, List[str]], str] = None,
        add_mlrun_logger: bool = True,
        mlrun_callback__kwargs: Dict[str, Any] = None,
        add_tensorboard_logger: bool = True,
        tensorboard_callback_kwargs: Dict[str, Any] = None,
    ):
        """
        Get automatic logging callbacks to both MLRun's context and Tensorboard. For further features of logging to both
        MLRun and Tensorboard, see 'pytorch.callbacks.MLRunLoggingCallback' and
        'pytorch.callbacks.TensorboardLoggingCallback'.

        :param custom_objects:              Custom objects the model is using. Expecting a dictionary with the classes
                                            names to import as keys (if multiple classes needed to be imported from the
                                            same py file a list can be given) and the python file from where to import
                                            them as their values. The model class itself must be specified in order to
                                            properly save it for later being loaded with a handler. For example:
                                            {
                                                "class_name": "/path/to/model.py",
                                                ["layer1", "layer2"]: "/path/to/custom_layers.py"
                                            }
        :param add_mlrun_logger:            Whether or not to add the 'MLRunLoggingCallback'. Defaulted to True.
        :param mlrun_callback__kwargs:      Key word arguments for the MLRun callback. For further information see the
                                            documentation of the class 'MLRunLoggingCallback'. Note that both 'context',
                                            'custom_objects' and 'auto_log' parameters are already given here.
        :param add_tensorboard_logger:      Whether or not to add the 'TensorboardLoggingCallback'. Defaulted to True.
        :param tensorboard_callback_kwargs: Key word arguments for the tensorboard callback. For further information see
                                            the documentation of the class 'TensorboardLoggingCallback'. Note that both
                                            'context' and 'auto_log' parameters are already given here.
        """
        # Set the dictionaries defaults:
        mlrun_callback__kwargs = (
            {} if mlrun_callback__kwargs is None else mlrun_callback__kwargs
        )
        tensorboard_callback_kwargs = (
            {} if tensorboard_callback_kwargs is None else tensorboard_callback_kwargs
        )

        # Add the loggers:
        if add_mlrun_logger:
            # Add the MLRun logging callback:
            self._callbacks.append(
                MLRunLoggingCallback(
                    context=self._context,
                    custom_objects=custom_objects,
                    auto_log=True,
                    **mlrun_callback__kwargs
                )
            )
        if add_tensorboard_logger:
            # Add the Tensorboard logging callback:
            self._callbacks.append(
                TensorboardLoggingCallback(
                    context=self._context, auto_log=True, **tensorboard_callback_kwargs
                )
            )

    def _parse_and_store(
        self,
        training_set: DataLoader = None,
        loss_function: Module = None,
        optimizer: Optimizer = None,
        validation_set: DataLoader = None,
        metric_functions: List[MetricFunctionType] = None,
        scheduler=None,
        scheduler_step_frequency: Union[int, float, str] = "epoch",
        epochs: int = 1,
        training_iterations: int = None,
        validation_iterations: int = None,
        callbacks: List[Callback] = None,
        use_cuda: bool = True,
        use_horovod: bool = None,
    ):
        """
        Parse and store the given input so the interface can starting training / evaluating.

        :param training_set:             A data loader for the training process.
        :param loss_function:            The loss function to use during training.
        :param optimizer:                The optimizer to use during the training.
        :param validation_set:           A data loader for the validation process.
        :param metric_functions:         The metrics to use on training and validation.
        :param scheduler:                Scheduler to use on the optimizer at the end of each epoch. The scheduler must
                                         have a 'step' method with no input.
        :param scheduler_step_frequency: The frequecny in which to step the given scheduler. Can be equal to one of the
                                         strings 'epoch' (for at the end of every epoch) and 'batch' (for at the end of
                                         every batch), or an integer that specify per how many iterations to step or a
                                         float percentage (0.0 < x < 1.0) for per x / iterations to step. Defaulted to
                                         'epoch'.
        :param epochs:                   Amount of epochs to perform. Defaulted to a single epoch.
        :param training_iterations:      Amount of iterations (batches) to perform on each epoch's training. If 'None'
                                         the entire training set will be used.
        :param validation_iterations:    Amount of iterations (batches) to perform on each epoch's validation. If 'None'
                                         the entire validation set will be used.
        :param callbacks:                The callbacks to use on this run.
        :param use_cuda:                 Whether or not to use cuda. Only relevant if cuda is available. Defaulted to
                                         True.
        :param use_horovod:              Whether or not to use horovod - a distributed training framework. Defaulted to
                                         None, meaning it will be read from context if available and if not - False.

        :raise ValueError: In case on of the given parameters is invalid.
        """
        # Parse and validate input:
        # # Metric functions:
        if metric_functions is None:
            metric_functions = []
        # # Training iterations:
        if training_set is not None:
            if training_iterations is None:
                training_iterations = len(training_set)
            elif training_iterations < 1:
                raise ValueError(
                    "The 'training_iterations' parameter must be bigger or equal to one, received: {}"
                    "".format(training_iterations)
                )
            elif training_iterations > len(training_set):
                raise ValueError(
                    "The 'training_iterations' cannot be bigger than the given training dataset. The size of "
                    "the given training set is {} yet the received iterations parameter is {}."
                    "".format(len(training_set), training_iterations)
                )
        # # Validation iterations:
        if validation_set is not None:
            if validation_iterations is None:
                validation_iterations = len(validation_set)
            elif validation_iterations < 1:
                raise ValueError(
                    "The 'validation_iterations' parameter must be bigger or equal to one, "
                    "received: {}".format(validation_iterations)
                )
            elif validation_iterations > len(validation_set):
                raise ValueError(
                    "The 'validation_iterations' cannot be bigger than the given validation dataset. The "
                    "size of the given validation set is {} yet the received iterations parameter is {}."
                    "".format(len(validation_set), validation_iterations)
                )
        # # Epochs:
        if epochs < 1:
            raise ValueError(
                "The 'epochs' parameter must be bigger or equal to one, received: {}".format(
                    epochs
                )
            )
        # # Scheduler step frequency:
        if isinstance(scheduler_step_frequency, str):
            if scheduler_step_frequency == "epoch":
                scheduler_step_frequency = training_iterations
            elif scheduler_step_frequency == "batch":
                scheduler_step_frequency = 1
            else:
                raise ValueError(
                    "The scheduler step frequency parameter can be passed as a string of two values: "
                    "'epoch' or 'batch', but the value given was: '{}'".format(
                        scheduler_step_frequency
                    )
                )
        elif isinstance(scheduler_step_frequency, float):
            if scheduler_step_frequency < 0.0 or scheduler_step_frequency > 1.0:
                raise ValueError(
                    "The scheduler step frequency parameter can be passed as a float with value between "
                    "0.0 to 1.0, but the value given was: '{}'".format(
                        scheduler_step_frequency
                    )
                )
            scheduler_step_frequency = int(
                training_iterations * scheduler_step_frequency
            )
        # # Callbacks:
        if callbacks is None:
            callbacks = []
        # # Use horovod:
        if use_horovod is None:
            use_horovod = (
                self._context.labels.get("kind", "") == "mpijob"
                if self._context is not None
                else False
            )

        # Store the configurations:
        self._training_set = training_set
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._validation_set = validation_set
        self._metric_functions = metric_functions
        self._scheduler = scheduler
        self._scheduler_step_frequency = scheduler_step_frequency
        self._epochs = epochs
        self._training_iterations = training_iterations
        self._validation_iterations = validation_iterations
        self._callbacks += callbacks
        self._use_cuda = use_cuda
        self._use_horovod = use_horovod

    def _objects_to_cuda(self):
        """
        Copy the interface objects - model, loss, optimizer and scheduler to cuda memory.
        """
        # Model:
        if not self._model_in_cuda:
            self._model = self._model.cuda()
            self._model_in_cuda = True

        # Loss:
        if self._loss_function is not None:
            self._loss_function = self._loss_function.cuda()

        # Optimizer:
        if self._optimizer is not None:
            for key in self._optimizer.state.keys():
                self._optimizer.state[key] = self._tensor_to_cuda(
                    tensor=self._optimizer.state[key]
                )

        # Scheduler:
        if self._scheduler is not None:
            for key in self._scheduler.__dict__.keys():
                self._scheduler.__dict__[key] = self._tensor_to_cuda(
                    tensor=self._scheduler.__dict__[key]
                )

    def _setup(self):
        """
        Setup the inner attributes of the interface, initializing horovod and the callbacks handler. This method must be
        called before train and evaluate.
        """
        # Setup horovod:
        if self._use_horovod:
            # Import horovod:
            self._hvd = importlib.import_module("horovod.torch")
            # Initialize horovod:
            self._hvd.init()
            # Limit the number of CPU threads to be used per worker:
            torch.set_num_threads(1)

        # Setup additional multiprocessing related key word arguments for the data loaders initialization:
        mp_data_loader_kwargs = {}

        # Setup cuda:
        if self._use_cuda and torch.cuda.is_available():
            if self._use_horovod:
                torch.cuda.set_device(self._hvd.local_rank())
                mp_data_loader_kwargs["num_workers"] = 1
                mp_data_loader_kwargs["pin_memory"] = True
            # Move the model and the stored objects to the GPU:
            self._objects_to_cuda()
        elif self._model_in_cuda:
            # Move the model back to the CPU:
            self._model = self._model.cpu()
            self._model_in_cuda = False

        # Initialize a callbacks handler:
        if self._use_horovod:
            self._callbacks_handler = CallbacksHandler(
                callbacks=[
                    callback
                    for callback in self._callbacks
                    if callback.on_horovod_check(rank=self._hvd.rank())
                ]
            )
        else:
            self._callbacks_handler = CallbacksHandler(callbacks=self._callbacks)

        # Prepare horovod for the run if needed:
        if self._use_horovod:
            # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent issues with
            # Infiniband implementations that are not fork-safe
            if (
                mp_data_loader_kwargs.get("num_workers", 0) > 0
                and hasattr(mp, "_supports_context")
                and mp._supports_context
                and "forkserver" in mp.get_all_start_methods()
            ):
                mp_data_loader_kwargs["multiprocessing_context"] = "forkserver"
            # Partition dataset among workers using distributed samplers:
            if self._training_set is not None:
                self._training_sampler = DistributedSampler(
                    self._training_set.dataset,
                    num_replicas=self._hvd.size(),
                    rank=self._hvd.rank(),
                )
                self._training_set = self._insert_sampler_to_data_loader(
                    data_loader=self._training_set,
                    sampler=self._training_sampler,
                    multiprocessing_kwargs=mp_data_loader_kwargs,
                )
            if self._validation_set is not None:
                self._validation_sampler = DistributedSampler(
                    self._validation_set.dataset,
                    num_replicas=self._hvd.size(),
                    rank=self._hvd.rank(),
                )
                self._validation_set = self._insert_sampler_to_data_loader(
                    data_loader=self._validation_set,
                    sampler=self._validation_sampler,
                    multiprocessing_kwargs=mp_data_loader_kwargs,
                )
            # Broadcast parameters and optimizer state:
            self._hvd.broadcast_parameters(self._model.state_dict(), root_rank=0)
            if self._optimizer is not None:
                self._hvd.broadcast_optimizer_state(self._optimizer, root_rank=0)
                # Add Horovod Distributed Optimizer:
                self._optimizer = self._hvd.DistributedOptimizer(
                    self._optimizer, named_parameters=self._model.named_parameters()
                )

        # Setup the callbacks functions:
        self._callbacks_handler.on_setup(
            model=self._model,
            training_set=self._training_set,
            validation_set=self._validation_set,
            loss_function=self._loss_function,
            optimizer=self._optimizer,
            metric_functions=self._metric_functions,
            scheduler=self._scheduler,
        )

    def _train(self):
        """
        Initiate a single epoch training.
        """
        # Set model to train mode:
        self._model.train()

        # Start the training:
        progress_bar = self._create_progress_bar(
            dataset=self._training_set,
            iterations=self._training_iterations,
            description="Training",
            metrics=[self._loss_function] + self._metric_functions,
        )
        for batch, (x, y_true) in progress_bar:
            # Check if iteration exceeded:
            if batch == self._training_iterations:
                break

            # Move to GPU if needed:
            if self._use_cuda and torch.cuda.is_available():
                x, y_true = self._tensor_to_cuda(tensor=(x, y_true))

            # Beginning of a batch callbacks:
            self._callbacks_handler.on_train_batch_begin(
                batch=batch, x=x, y_true=y_true
            )

            # Zero the parameters gradients:
            self._optimizer.zero_grad()

            # Infer the input:
            self._callbacks_handler.on_inference_begin(x=x)
            y_pred = self._model(x)
            self._callbacks_handler.on_inference_end(y_pred=y_pred, y_true=y_true)

            # Calculate loss:
            self._callbacks_handler.on_train_loss_begin()
            loss_value = self._loss_function(y_pred, y_true)
            self._callbacks_handler.on_train_loss_end(loss_value=loss_value)

            # Measure accuracies:
            self._callbacks_handler.on_train_metrics_begin()
            metric_values = self._metrics(y_pred=y_pred, y_true=y_true)
            self._callbacks_handler.on_train_metrics_end(metric_values=metric_values)

            # Update the progress bar with the recent values:
            self._update_progress_bar(
                progress_bar=progress_bar,
                metrics=[self._loss_function] + self._metric_functions,
                values=[loss_value] + metric_values,
            )

            # Perform backward propagation:
            self._callbacks_handler.on_backward_begin()
            loss_value.backward()
            self._callbacks_handler.on_backward_end()

            # Step optimizer:
            self._callbacks_handler.on_optimizer_step_begin()
            self._optimizer.step()
            self._callbacks_handler.on_optimizer_step_end()

            # Step scheduler:
            if (
                self._scheduler is not None
                and (batch + 1) % self._scheduler_step_frequency == 0
            ):
                self._callbacks_handler.on_scheduler_step_begin()
                self._scheduler.step()
                self._callbacks_handler.on_scheduler_step_end()

            # End of batch callbacks:
            if not self._callbacks_handler.on_train_batch_end(
                batch=batch, x=x, y_pred=y_pred, y_true=y_true
            ):
                break

    def _validate(
        self, is_evaluation: bool = False
    ) -> Tuple[MetricValueType, List[MetricValueType]]:
        """
        Initiate a single epoch validation.

        :param is_evaluation: Whether or not this call is part of an evaluation or training.

        :return: A tuple of the validation summary:
                 [0] = Validation loss value summary.
                 [1] = A list of metrics summaries.
        """
        # Set model to evaluate mode:
        self._model.eval()

        # Start the validation:
        losses = []
        metrics = []
        progress_bar = self._create_progress_bar(
            dataset=self._validation_set,
            iterations=self._validation_iterations,
            description="Evaluating" if is_evaluation else "Validating",
            metrics=[self._loss_function] + self._metric_functions,
        )
        with torch.no_grad():
            for batch, (x, y_true) in progress_bar:
                # Check if iteration exceeded:
                if batch == self._validation_iterations:
                    break

                # Move to GPU if needed:
                if self._use_cuda and torch.cuda.is_available():
                    x, y_true = self._tensor_to_cuda(tensor=(x, y_true))

                # Beginning of a batch callbacks:
                self._callbacks_handler.on_validation_batch_begin(
                    batch=batch, x=x, y_true=y_true
                )

                # Infer the input:
                self._callbacks_handler.on_inference_begin(x=x)
                y_pred = self._model(x)
                self._callbacks_handler.on_inference_end(y_pred=y_pred, y_true=y_true)

                # Calculate loss:
                self._callbacks_handler.on_validation_loss_begin()
                loss_value = self._loss_function(y_pred, y_true)
                self._callbacks_handler.on_validation_loss_end(loss_value=loss_value)

                # Measure accuracies:
                self._callbacks_handler.on_validation_metrics_begin()
                metric_values = self._metrics(y_pred=y_pred, y_true=y_true)
                self._callbacks_handler.on_validation_metrics_end(
                    metric_values=metric_values
                )

                # Update the progress bar with the recent values:
                self._update_progress_bar(
                    progress_bar=progress_bar,
                    metrics=[self._loss_function] + self._metric_functions,
                    values=[loss_value] + metric_values,
                )

                # Collect results:
                losses.append(loss_value)
                metrics.append(metric_values)

                # End of batch callbacks:
                if not self._callbacks_handler.on_validation_batch_end(
                    batch=batch, x=x, y_pred=y_pred, y_true=y_true,
                ):
                    break

        # Calculate the final average of the loss and accuracy values:
        loss_value = sum(losses) / len(losses)
        metric_values = (
            [(sum(metric) / len(metric)) for metric in metrics]
            if len(metrics) > 0
            else []
        )
        return loss_value, metric_values

    def _print_results(self, loss_value: Tensor, metric_values: List[float]):
        """
        Print the given result between each epoch.

        :param loss_value:    The loss result to print.
        :param metric_values: The metrics result to print.
        """
        table = [[self._get_metric_name(metric=self._loss_function), float(loss_value)]]
        for metric_function, metric_value in zip(self._metric_functions, metric_values):
            table.append([self._get_metric_name(metric=metric_function), metric_value])
        print(
            "\nSummary:\n"
            + tabulate(table, headers=["Metrics", "Values"], tablefmt="pretty")
        )

    def _metrics(self, y_pred: Tensor, y_true: Tensor) -> List[float]:
        """
        Call all the metrics on the given batch's truth and prediction output.

        :param y_pred: The batch's truth value.
        :param y_true: The model's output for the related input.

        :return: A list with each metric result.
        """
        accuracies = []
        for metric_function in self._metric_functions:
            accuracies.append(metric_function(y_pred, y_true))
        return accuracies

    def _metric_average(self, rank_value: Union[Tensor, float], name: str) -> float:
        """
        Wait for all ranks and calculate the average of the metric provided.

        :param rank_value: The caller rank metric value.
        :param name:       The metric name.

        :return: The metric average across all ranks.
        """
        if not isinstance(rank_value, Tensor):
            rank_value = torch.tensor(rank_value)
        average_tensor = self._hvd.allreduce(rank_value, name=name)
        return average_tensor.item()

    def _get_learning_rate(self) -> Union[Tuple[str, List[Union[str, int]]], None]:
        """
        Try and get the learning rate value form the stored optimizer.

        :return: The key chain to get the optimizer learning rate value or None if the learning rate could not be
                 accessed via the common key.
        """
        if "lr" in self._optimizer.param_groups[0]:
            return HyperparametersKeys.OPTIMIZER, ["param_groups", 0, "lr"]
        if "learning_rate" in self._optimizer.param_groups[0]:
            return HyperparametersKeys.OPTIMIZER, ["param_groups", 0, "learning_rate"]
        return None

    def _clear(self):
        """
        Clear the interface from the methods parameters, setting them back to None. The interface's model will not be
        touched.
        """
        # Clear the methods parameters:
        self._training_set = None  # type: DataLoader
        self._loss_function = None  # type: Module
        self._optimizer = None  # type: Optimizer
        self._validation_set = None  # type: DataLoader
        self._metric_functions = None  # type: List[MetricFunctionType]
        self._scheduler = None
        self._scheduler_step_frequency = None  # type: int
        self._epochs = None  # type: int
        self._training_iterations = None  # type: int
        self._validation_iterations = None  # type: int
        self._callbacks = []  # type: List[Callback]
        self._use_cuda = None  # type: bool
        self._use_horovod = None  # type: bool

        # Clear the inner attributes:
        self._hvd = None

        self._training_sampler = None  # type: DistributedSampler
        self._validation_sampler = None  # type: DistributedSampler
        self._callbacks_handler = None  # type: CallbacksHandler

    @staticmethod
    def _insert_sampler_to_data_loader(
        data_loader: DataLoader,
        sampler: DistributedSampler,
        multiprocessing_kwargs: dict,
    ):
        """
        Initialize a new data loader based on the given data loader with the given sampler.

        :param data_loader:            The data loader to insert the sampler to.
        :param sampler:                Sampler to insert.
        :param multiprocessing_kwargs: Additional keyword arguments for the multiprocessing attributes.

        :return: The sampler initialized data loader.
        """
        pin_memory = multiprocessing_kwargs.get("pin_memory", data_loader.pin_memory)
        num_workers = multiprocessing_kwargs.get("num_workers", data_loader.num_workers)
        multiprocessing_context = multiprocessing_kwargs.get(
            "multiprocessing_context", data_loader.multiprocessing_context
        )
        with_sampler_data_loader = DataLoader(
            dataset=data_loader.dataset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=pin_memory,
            drop_last=data_loader.drop_last,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
        )
        del data_loader
        return with_sampler_data_loader

    @staticmethod
    def _tensor_to_cuda(
        tensor: Union[Tensor, Dict, List, Tuple]
    ) -> Union[Tensor, Dict, List, Tuple]:
        """
        Send to given tensor to cuda if it is a tensor. If the given object is a dictionary, the dictionary values will
        be sent to the function again recursively. If the given object is a list or a tuple, all the values in it will
        be sent as well. If the given object is not of type torch.Tensor at the end, nothing will happen.

        :param tensor: The batch to sent to cuda.

        :return: The copied tensor in cuda memory.
        """
        if isinstance(tensor, Tensor):
            tensor = tensor.cuda()
            if tensor._grad is not None:
                tensor._grad.data = tensor._grad.data.cuda()
        elif isinstance(tensor, dict):
            for key in tensor:
                tensor[key] = PyTorchMLRunInterface._tensor_to_cuda(tensor=tensor[key])
        elif isinstance(tensor, list):
            for index in range(len(tensor)):
                tensor[index] = PyTorchMLRunInterface._tensor_to_cuda(
                    tensor=tensor[index]
                )
        elif isinstance(tensor, tuple):
            cuda_tensor = ()
            for value in tensor:
                cuda_tensor += (PyTorchMLRunInterface._tensor_to_cuda(tensor=value),)
            tensor = cuda_tensor
        return tensor

    @staticmethod
    def _get_metric_name(metric: MetricFunctionType) -> str:
        """
        Get the given metric function name.

        :param metric: The metric function pointer.

        :return: The metric name.
        """
        if isinstance(metric, Module):
            return metric.__class__.__name__
        return metric.__name__

    @staticmethod
    def _create_progress_bar(
        dataset: DataLoader,
        iterations: int,
        description: str,
        metrics: List[MetricFunctionType],
    ) -> tqdm:
        """
        Create a progress bar for training and validating / evaluating.

        :param dataset:     The dataset to enumerate in this progress bar.
        :param iterations:  The amount of iterations that will be performed.
        :param description: The header appearing in the left most side of the progress bar.
        :param metrics:     Metrics to note in the right most side of the progress bar.

        :return: The created progress bar.
        """
        return tqdm(
            iterable=enumerate(dataset),
            bar_format="{desc}: {percentage:3.0f}%"
            " |{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            desc=description,
            postfix={
                PyTorchMLRunInterface._get_metric_name(metric=metric): "?"
                for metric in metrics
            },
            unit="Batch",
            total=iterations,
            ascii=False,
            file=sys.stdout,
        )

    @staticmethod
    def _update_progress_bar(
        progress_bar: tqdm,
        metrics: List[MetricFunctionType],
        values: List[MetricValueType],
    ):
        """
        Update the progress bar metrics results.

        :param progress_bar: The progress bar to update.
        :param metrics:      The metrics list noted in the progress bar.
        :param values:       The metrics recent calculated values.
        """
        progress_bar.set_postfix(
            ordered_dict={
                PyTorchMLRunInterface._get_metric_name(metric=metric): float(value)
                for metric, value in zip(metrics, values)
            },
            refresh=False,
        )
