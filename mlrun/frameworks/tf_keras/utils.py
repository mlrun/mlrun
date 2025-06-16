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
import tensorflow as tf
from packaging import version
from tensorflow import keras

import mlrun
from mlrun.data_types import ValueType

from .._dl_common import DLTypes, DLUtils


class TFKerasTypes(DLTypes):
    """
    Typing hints for the TensorFlow.Keras framework.
    """

    # Every model in tf.keras must inherit from tf.keras.Model:
    ModelType = keras.Model


class TFKerasUtils(DLUtils):
    """
    Utilities functions for the TensorFlow.Keras framework.
    """

    @staticmethod
    def convert_value_type_to_tf_dtype(
        value_type: str,
    ) -> tf.DType:
        """
        Get the 'tensorflow.DType' equivalent to the given MLRun value type.

        :param value_type: The MLRun value type to convert to tensorflow data type.

        :return: The 'tensorflow.DType' equivalent to the given MLRun data type.

        :raise MLRunInvalidArgumentError: If tensorflow is not supporting the given data type.
        """
        # Initialize the mlrun to tensorflow data type conversion map:
        conversion_map = {
            ValueType.BOOL: tf.bool,
            ValueType.INT8: tf.int8,
            ValueType.INT16: tf.int16,
            ValueType.INT32: tf.int32,
            ValueType.INT64: tf.int64,
            ValueType.UINT8: tf.uint8,
            ValueType.UINT16: tf.uint16,
            ValueType.UINT32: tf.uint32,
            ValueType.UINT64: tf.uint64,
            ValueType.BFLOAT16: tf.bfloat16,
            ValueType.FLOAT16: tf.float16,
            ValueType.FLOAT: tf.float32,
            ValueType.DOUBLE: tf.float64,
        }

        # Convert and return:
        if value_type in conversion_map:
            return conversion_map[value_type]
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The ValueType given is not supported in tensorflow: '{value_type}'."
        )

    @staticmethod
    def convert_tf_dtype_to_value_type(
        tf_dtype: tf.DType,
    ) -> str:
        """
        Convert the given tensorflow data type to MLRun data type. All the CUDA supported data types are supported.
        For more information regarding tensorflow data types,
        visit: https://www.tensorflow.org/api_docs/python/tf/dtypes

        :param tf_dtype: The tensorflow data type to convert to MLRun's data type. Expected to be a 'tensorflow.dtype'
                         or 'str'.

        :return: The MLRun value type converted from the given data type.

        :raise MLRunInvalidArgumentError: If the tensorflow data type is not supported by MLRun.
        """
        # Initialize the tensorflow to mlrun data type conversion map:
        conversion_map = {
            tf.bool.name: ValueType.BOOL,
            tf.int8.name: ValueType.INT8,
            tf.int16.name: ValueType.INT16,
            tf.int32.name: ValueType.INT32,
            tf.int64.name: ValueType.INT64,
            tf.uint8.name: ValueType.UINT8,
            tf.uint16.name: ValueType.UINT16,
            tf.uint32.name: ValueType.UINT32,
            tf.uint64.name: ValueType.UINT64,
            tf.bfloat16.name: ValueType.BFLOAT16,
            tf.half.name: ValueType.FLOAT16,
            tf.float16.name: ValueType.FLOAT16,
            tf.float32.name: ValueType.FLOAT,
            tf.double.name: ValueType.DOUBLE,
            tf.float64.name: ValueType.DOUBLE,
        }

        # Parse the given tensorflow data type to string:
        if isinstance(tf_dtype, tf.DType):
            tf_dtype = tf_dtype.name

        # Convert and return:
        if tf_dtype in conversion_map:
            return conversion_map[tf_dtype]
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"MLRun value type is not supporting the given tensorflow data type: '{tf_dtype}'."
        )


def is_keras_3() -> bool:
    """
    Check if the current Keras version is 3.x.

    :return: True if Keras version is 3.x, False otherwise.
    """
    return hasattr(keras, "__version__") and version.parse(
        keras.__version__
    ) >= version.parse("3.0.0")
