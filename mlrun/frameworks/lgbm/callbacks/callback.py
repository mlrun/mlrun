from lightgbm.callback import CallbackEnv

from abc import ABC, abstractmethod


class Callback(ABC):
    """
    A callback class for writing callbacks usable by lightgbm's train function. The function expects a callable object,
    as such the '__call__` must be implemented.

    'lgb.train' will pass a CallbackEnv object to the callbacks - a NamedTuple with the following items:

    * model: Booster - The model's Booster object.
    * params: Dict - The training parameters passed to 'lgb.train'.
    * iteration: int - The current iteration.
    * begin_iteration: int - The first iteration.
    * end_iteration: int - The last iteration.
    * evaluation_result_list: List[Tuple[str, str, float, bool]] - A list of metric tuples. Each metric tuple is
      constructed by:
      [0] str - The validation set name the metric was calculated on.
      [1] str - The metric name.
      [2] float - The metric score.
      [3] bool - The aim of the metric, True means bigger score is better and False means smaller score is better.
    """

    @abstractmethod
    def __call__(self, env: CallbackEnv) -> None:
        """
        The function to be called by the training at the end of each iteration (post validating on all the given
        validation datasets).

        :param env: The CallbackEnv representing the current iteration of the training.
        """
        pass

