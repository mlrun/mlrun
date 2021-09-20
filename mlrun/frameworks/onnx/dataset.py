from typing import Callable, List, Union

import numpy as np


class ONNXDataset:
    # TODO: Finish this
    def __init__(
        self,
        x: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]] = None,
        batch_size: int = 1,
        transforms: List[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._transforms = transforms
