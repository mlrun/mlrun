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

from lightgbm.callback import CallbackEnv


class Callback(ABC):
    """
    An abstract callback class for writing callbacks usable by LightGBM's training. The function expects a callable
    object, as such the `__call__` method will be used.

    There are two configurable class properties:

    * order: int = 10 - The priority of the callback to be called first. Lower value means higher priority. Default:
      10.
    * before_iteration: bool = False - Whether to call this callback before each iteration or after. Default: after
      (False).

    LightGBM will pass a `CallbackEnv` object to the callbacks - a `NamedTuple` with the following items:

    * model: Booster - The model's Booster object.
    * params: Dict - The training parameters passed to 'lgb.train'.
    * iteration: int - The current iteration.
    * begin_iteration: int - The first iteration.
    * end_iteration: int - The last iteration.
    * evaluation_result_list: List[Tuple[str, str, float, bool, Optional[float]]] - A list of metric tuples. Each metric
      tuple is constructed by:

      * [0] str - The validation set name the metric was calculated on.
      * [1] str - The metric name.
      * [2] float - The metric score (mean score in case of `lightgbm.cv`).
      * [3] bool - The aim of the metric, True means bigger score is better and False means smaller score is better.
      * [4] Optional[float] - The metric stdv score (only in case of `lightgbm.cv`)

    In addition, if the LightGBM module or model are wrapped with MLRun, the methods `on_train_begin` and `on_train_end`
    will be called as well.

    example::

        class ExampleCallback(Callback):
            def __init__(self, name: str):
                self.name = name

            def __call__(self, env: CallbackEnv):
                print(f"{self.name}: current iteration: {env.iteration}")

            def on_train_begin(self):
                print("{self.name}: Starting training...")

            def on_train_end(self):
                print("{self.name}: Done training!")


        apply_mlrun()
        lgb.train(..., callbacks=[ExampleCallback(name="Example")])
    """

    def __init__(self, order: int = 10, before_iteration: bool = False):
        """
        Initialize a new callback to use in LightGBM's training.

        :param order:            The priority of the callback to be called first. Lower value means higher priority.
                                 Default: 10.
        :param before_iteration: Whether to call this callback before each iteration or after. Default: after
                                 (False).
        """
        self.order = order
        self.before_iteration = before_iteration

    def __call__(self, env: CallbackEnv):
        """
        The method to be called during training of a LightGBM model. It will be called at the end of each iteration
        post validating on all the given validation datasets.

        :param env: The CallbackEnv representing the current iteration of the training.
        """
        pass

    def on_train_begin(self):
        """
        Method to be called before the training starts. Will only be called if the model is wrapped with an MLRun
        interface.
        """
        pass

    def on_train_end(self):
        """
        Method to be called after the training ends. Will only be called if the model is wrapped with an MLRun
        interface.
        """
        pass
