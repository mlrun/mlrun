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
import math
from typing import Callable, Optional, Union

import numpy as np


class ONNXDataset:
    """
    An iterable class handling numpy arrays for ONNX inference.
    """

    def __init__(
        self,
        x: Union[np.ndarray, list[np.ndarray]],
        y: Union[np.ndarray, list[np.ndarray]] = None,
        batch_size: int = 1,
        x_transforms: Optional[list[Callable[[np.ndarray], np.ndarray]]] = None,
        y_transforms: Optional[list[Callable[[np.ndarray], np.ndarray]]] = None,
        is_batched_transforms: bool = False,
    ):
        """
        Initialize a new ONNX dataset.

        :param x:                     The data to handle.
        :param y:                     Ground truth for the data 'x'.
        :param batch_size:            The batch size to use. Default: 1.
        :param x_transforms:          A list of callable transforms to apply on the 'x' data.
        :param y_transforms:          A list of callable transforms to apply on the 'y' ground truth.
        :param is_batched_transforms: Whether the given transforms support batched data or not. If not, the transform
                                      will be applied on each item of a given batch. Default: False.
        """
        # Store given parameters:
        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._x_transforms = x_transforms
        self._y_transforms = y_transforms
        self._is_batched_transforms = is_batched_transforms

        # Set inner index to 0:
        self._index = 0

    def __len__(self):
        """
        Calculate the dataset's length in batches.

        :return: The dataset's length.
        """
        return math.ceil(len(self._x) / self._batch_size)

    def __iter__(self):
        """
        Set the inner index of the dataset to 0 and return the dataset as the iterator.

        :return: The dataset itself.
        """
        self._index = 0
        return self

    def __next__(self) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Get the next item in line (by the inner index) since calling '__iter__'. If ground truth was provided (y),
        a tuple of (x, y) will be returned. Otherwise x.

        :return: The next item.

        :raise StopIteration: If there are no more items to be returned.
        """
        # Validate the next item is in range:
        if self._index >= len(self):
            raise StopIteration

        # Get the next item:
        item = self[self._index]
        self._index += 1

        return item

    def __getitem__(
        self, index: int
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Get the item at the given index. If ground truth was provided, a tuple of (x, y) will be returned. Otherwise x.

        :param index: The index to get the item. Must be in range of 0 to dataset's length.

        :return: The item at the given index.

        :raise IndexError: If the given index is not in range.
        """
        # Calculate the indexes:
        from_index = self._batch_size * index
        if from_index >= len(self):
            raise IndexError(
                f"Dataset's length is {len(self)}, yet given index was {index}."
            )
        to_index = max(self._batch_size * (index + 1), len(self._x))
        is_batched = (to_index - from_index) > 1

        # Get the index data:
        x = self._x[from_index:to_index]
        x = self._call_x_transforms(x=x, is_batched=is_batched)
        if self._y is None:
            return x

        # Get the index ground truth:
        y = self._y[from_index:to_index]
        y = self._call_y_transforms(y=y, is_batched=is_batched)
        return x, y

    def _call_x_transforms(self, x: np.ndarray, is_batched: bool):
        """
        Call the data transforms on the given input.

        :param x:          The input to get trough the data transforms.
        :param is_batched: Whether or not the given input is batched.

        :return: Transformed input.
        """
        if self._x_transforms is not None:
            return self._call_transforms(
                items=x, transforms=self._x_transforms, is_batched=is_batched
            )
        return x

    def _call_y_transforms(self, y: np.ndarray, is_batched: bool):
        """
        Call the ground truth transforms on the given input.

        :param y:          The input to get trough the ground truth transforms.
        :param is_batched: Whether or not the given input is batched.

        :return: Transformed input.
        """
        if self._y_transforms is not None:
            return self._call_transforms(
                items=y, transforms=self._y_transforms, is_batched=is_batched
            )
        return y

    def _call_transforms(
        self,
        items: np.ndarray,
        transforms: list[Callable[[np.ndarray], np.ndarray]],
        is_batched: bool,
    ):
        """
        Propagate the given items through the given transforms.

        :param items:      The items to transform.
        :param transforms: The transforms to use.
        :param is_batched: Whether or not the items are in a batch or not (single item).

        :return: The transformed items.
        """
        if self._is_batched_transforms or not is_batched:
            for transform in transforms:
                items = transform(items)
        else:
            for transform in transforms:
                items = [transform(item) for item in items]

        return items
