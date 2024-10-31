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
from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

import mlrun
from mlrun.data_types import ValueType

from .._dl_common import DLTypes, DLUtils


class PyTorchTypes(DLTypes):
    """
    Typing hints for the PyTorch framework.
    """

    # Every model in PyTorch must inherit from torch.nn.Module:
    ModelType = Module

    # Supported types of loss and metrics values:
    MetricValueType = Union[int, float, np.ndarray, Tensor]

    # Supported types of metrics:
    MetricFunctionType = Union[Callable[[Tensor, Tensor], MetricValueType], Module]


class PyTorchUtils(DLUtils):
    """
    Utilities functions for the PyTorch framework.
    """

    @staticmethod
    def convert_value_type_to_torch_dtype(
        value_type: str,
    ) -> torch.dtype:
        """
        Get the 'torch.dtype' equivalent to the given MLRun data type.

        :param value_type: The MLRun value type to convert to torch data type.

        :return: The 'torch.dtype' equivalent to the given MLRun data type.

        :raise MLRunInvalidArgumentError: If torch is not supporting the given value type.
        """
        # Initialize the mlrun to torch data type conversion map:
        conversion_map = {
            ValueType.BOOL: torch.bool,
            ValueType.INT8: torch.int8,
            ValueType.INT16: torch.int16,
            ValueType.INT32: torch.int32,
            ValueType.INT64: torch.int64,
            ValueType.UINT8: torch.uint8,
            ValueType.BFLOAT16: torch.bfloat16,
            ValueType.FLOAT16: torch.float16,
            ValueType.FLOAT: torch.float32,
            ValueType.DOUBLE: torch.float64,
        }

        # Convert and return:
        if value_type in conversion_map:
            return conversion_map[value_type]
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The ValueType given is not supported in torch: '{value_type}'."
        )

    @staticmethod
    def convert_torch_dtype_to_value_type(torch_dtype: Union[torch.dtype, str]) -> str:
        """
        Convert the given torch data type to MLRun value type. All the CUDA supported data types are supported. For
        more information regarding torch data types, visit: https://pytorch.org/docs/stable/tensors.html#data-types

        :param torch_dtype: The torch data type to convert to MLRun's value type. Expected to be a 'torch.dtype' or
                            'str'.

        :return: The MLRun value type converted from the given data type.

        :raise MLRunInvalidArgumentError: If the torch data type is not supported by MLRun.
        """
        # Initialize the torch to mlrun data type conversion map:
        conversion_map = {
            str(torch.bool): ValueType.BOOL,
            str(torch.int8): ValueType.INT8,
            str(torch.short): ValueType.INT16,
            str(torch.int16): ValueType.INT16,
            str(torch.int): ValueType.INT32,
            str(torch.int32): ValueType.INT32,
            str(torch.long): ValueType.INT64,
            str(torch.int64): ValueType.INT64,
            str(torch.uint8): ValueType.UINT8,
            str(torch.bfloat16): ValueType.BFLOAT16,
            str(torch.half): ValueType.FLOAT16,
            str(torch.float16): ValueType.FLOAT16,
            str(torch.float): ValueType.FLOAT,
            str(torch.float32): ValueType.FLOAT,
            str(torch.double): ValueType.DOUBLE,
            str(torch.float64): ValueType.DOUBLE,
        }

        # Parse the given torch data type to string:
        if isinstance(torch_dtype, torch.dtype):
            torch_dtype = str(torch_dtype)

        # Convert and return:
        if torch_dtype in conversion_map:
            return conversion_map[torch_dtype]
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"MLRun value type is not supporting the given torch data type: '{torch_dtype}'."
        )
