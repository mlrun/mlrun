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
import os
import pathlib
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from mlrun.artifacts import Artifact, DatasetArtifact
from mlrun.datastore import DataItem
from mlrun.errors import MLRunInvalidArgumentError

from ..utils import ArtifactType, SupportedFormat
from .default_packager import DefaultPackager

# Type for collection of numpy arrays (list / dict of arrays):
NumPyArrayCollectionType = Union[list[np.ndarray], dict[str, np.ndarray]]


class _Formatter(ABC):
    """
    An abstract class for a numpy formatter - supporting saving and loading arrays to and from specific file type.
    """

    @classmethod
    @abstractmethod
    def save(
        cls,
        obj: Union[np.ndarray, NumPyArrayCollectionType],
        file_path: str,
        **save_kwargs: dict,
    ):
        """
        Save the given array to the file path given.

        :param obj:         The numpy array to save.
        :param file_path:   The file to save to.
        :param save_kwargs: Additional keyword arguments to pass to the relevant save function of numpy.
        """
        pass

    @classmethod
    @abstractmethod
    def load(
        cls, file_path: str, **load_kwargs: dict
    ) -> Union[np.ndarray, NumPyArrayCollectionType]:
        """
        Load the array from the given file path.

        :param file_path:   The file to load the array from.
        :param load_kwargs: Additional keyword arguments to pass to the relevant load function of numpy.

        :return: The loaded array.
        """
        pass


class _TXTFormatter(_Formatter):
    """
    A static class for managing numpy txt files.
    """

    @classmethod
    def save(cls, obj: np.ndarray, file_path: str, **save_kwargs: dict):
        """
        Save the given array to the file path given.

        :param obj:         The numpy array to save.
        :param file_path:   The file to save to.
        :param save_kwargs: Additional keyword arguments to pass to the relevant save function of numpy.

        :raise MLRunInvalidArgumentError: If the array is above 2D.
        """
        if len(obj.shape) > 2:
            raise MLRunInvalidArgumentError(
                f"Cannot save the given array to file. Only 1D and 2D arrays can be saved to text files but the given "
                f"array is {len(obj.shape)}D (shape of {obj.shape}). Please use 'npy' format instead."
            )
        np.savetxt(file_path, obj, **save_kwargs)

    @classmethod
    def load(cls, file_path: str, **load_kwargs: dict) -> np.ndarray:
        """
        Load the array from the given 'txt' file path.

        :param file_path:   The file to load the array from.
        :param load_kwargs: Additional keyword arguments to pass to the relevant load function of numpy.

        :return: The loaded array.
        """
        return np.loadtxt(file_path, **load_kwargs)


class _CSVFormatter(_TXTFormatter):
    """
    A static class for managing numpy csv files.
    """

    @classmethod
    def save(cls, obj: np.ndarray, file_path: str, **save_kwargs: dict):
        """
        Save the given array to the file path given.

        :param obj:         The numpy array to save.
        :param file_path:   The file to save to.
        :param save_kwargs: Additional keyword arguments to pass to the relevant save function of numpy.

        :raise MLRunInvalidArgumentError: If the array is above 2D.
        """
        super().save(obj=obj, file_path=file_path, **{"delimiter": ",", **save_kwargs})

    @classmethod
    def load(cls, file_path: str, **load_kwargs: dict) -> np.ndarray:
        """
        Load the array from the given 'txt' file path.

        :param file_path:   The file to load the array from.
        :param load_kwargs: Additional keyword arguments to pass to the relevant load function of numpy.

        :return: The loaded array.
        """
        return super().load(file_path=file_path, **{"delimiter": ",", **load_kwargs})


class _NPYFormatter(_Formatter):
    """
    A static class for managing numpy npy files.
    """

    @classmethod
    def save(cls, obj: np.ndarray, file_path: str, **save_kwargs: dict):
        """
        Save the given array to the file path given.

        :param obj:         The numpy array to save.
        :param file_path:   The file to save to.
        :param save_kwargs: Additional keyword arguments to pass to the relevant save function of numpy.
        """
        np.save(file_path, obj, **save_kwargs)

    @classmethod
    def load(cls, file_path: str, **load_kwargs: dict) -> np.ndarray:
        """
        Load the array from the given 'npy' file path.

        :param file_path:   The file to load the array from.
        :param load_kwargs: Additional keyword arguments to pass to the relevant load function of numpy.

        :return: The loaded array.
        """
        return np.load(file_path, **load_kwargs)


class _NPZFormatter(_Formatter):
    """
    A static class for managing numpy npz files.
    """

    @classmethod
    def save(
        cls,
        obj: NumPyArrayCollectionType,
        file_path: str,
        is_compressed: bool = False,
        **save_kwargs: dict,
    ):
        """
        Save the given array to the file path given.

        :param obj:           The numpy array to save.
        :param file_path:     The file to save to.
        :param is_compressed: Whether to save it as a compressed npz file.
        :param save_kwargs:   Additional keyword arguments to pass to the relevant save function of numpy.
        """
        save_function = np.savez_compressed if is_compressed else np.savez
        if isinstance(obj, list):
            save_function(file_path, *obj)
        else:
            save_function(file_path, **obj)

    @classmethod
    def load(cls, file_path: str, **load_kwargs: dict) -> dict[str, np.ndarray]:
        """
        Load the arrays from the given 'npz' file path.

        :param file_path:   The file to load the array from.
        :param load_kwargs: Additional keyword arguments to pass to the relevant load function of numpy.

        :return: The loaded arrays as a mapping (dictionary) of type `np.lib.npyio.NpzFile`.
        """
        return np.load(file_path, **load_kwargs)


class NumPySupportedFormat(SupportedFormat[_Formatter]):
    """
    Library of numpy formats (file extensions) supported by the NumPy packagers.
    """

    NPY = "npy"
    NPZ = "npz"
    TXT = "txt"
    GZ = "gz"
    CSV = "csv"

    _FORMAT_HANDLERS_MAP = {
        NPY: _NPYFormatter,
        NPZ: _NPZFormatter,
        TXT: _TXTFormatter,
        GZ: _TXTFormatter,  # 'gz' format handled the same as 'txt'.
        CSV: _CSVFormatter,
    }

    @classmethod
    def get_single_array_formats(cls) -> list[str]:
        """
        Get the supported formats for saving one numpy array.

        :return: A list of all the supported formats for saving one numpy array.
        """
        return [cls.NPY, cls.TXT, cls.GZ, cls.CSV]

    @classmethod
    def get_multi_array_formats(cls) -> list[str]:
        """
        Get the supported formats for saving a collection (multiple) numpy arrays - e.g. list of arrays or dictionary of
        arrays.

        :return: A list of all the supported formats for saving multiple numpy arrays.
        """
        return [cls.NPZ]


# Default file formats for numpy arrays file artifacts:
DEFAULT_NUMPY_ARRAY_FORMAT = NumPySupportedFormat.NPY
DEFAULT_NUMPPY_ARRAY_COLLECTION_FORMAT = NumPySupportedFormat.NPZ


class NumPyNDArrayPackager(DefaultPackager):
    """
    ``numpy.ndarray`` packager.
    """

    PACKABLE_OBJECT_TYPE = np.ndarray

    # The size of an array to be stored as a result, rather than a file in the `get_default_packing_artifact_type`
    # method:
    _ARRAY_SIZE_AS_RESULT = 10

    def get_default_packing_artifact_type(self, obj: np.ndarray) -> str:
        """
        Get the default artifact type. Will be a result if the array size is less than 10, otherwise file.

        :param obj: The about to be packed array.

        :return: The default artifact type.
        """
        if obj.size < self._ARRAY_SIZE_AS_RESULT:
            return ArtifactType.RESULT
        return ArtifactType.FILE

    def get_default_unpacking_artifact_type(self, data_item: DataItem) -> str:
        """
        Get the default artifact type used for unpacking. Returns dataset if the data item represents a
        `DatasetArtifact` and otherwise, file.

        :param data_item: The about to be unpacked data item.

        :return: The default artifact type.
        """
        is_artifact = data_item.get_artifact_type()
        if is_artifact and is_artifact == "datasets":
            return ArtifactType.DATASET
        return ArtifactType.FILE

    def pack_result(self, obj: np.ndarray, key: str) -> dict:
        """
        Pack an array as a result.

        :param obj: The array to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        # If the array is a number (size of 1), then we'll lok it as a single number. Otherwise, log as a list result:
        if obj.size == 1:
            obj = obj.item()
        else:
            obj = obj.tolist()

        return super().pack_result(obj=obj, key=key)

    def pack_file(
        self,
        obj: np.ndarray,
        key: str,
        file_format: str = DEFAULT_NUMPY_ARRAY_FORMAT,
        **save_kwargs,
    ) -> tuple[Artifact, dict]:
        """
        Pack an array as a file by the given format.

        :param obj:         The aray to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is npy.
        :param save_kwargs: Additional keyword arguments to pass to the numpy save functions.

        :return: The packed artifact and instructions.
        """
        # Save to file:
        formatter = NumPySupportedFormat.get_format_handler(fmt=file_format)
        temp_directory = pathlib.Path(tempfile.mkdtemp())
        self.add_future_clearing_path(path=temp_directory)
        file_path = temp_directory / f"{key}.{file_format}"
        formatter.save(obj=obj, file_path=str(file_path), **save_kwargs)

        # Create the artifact and instructions (Note: only 'npy' format support saving object arrays and that will
        # require pickling, hence we set the required instruction):
        artifact = Artifact(key=key, src_path=os.path.abspath(file_path))
        instructions = {"file_format": file_format}
        if file_format == NumPySupportedFormat.NPY and obj.dtype == np.object_:
            instructions["allow_pickle"] = True

        return artifact, instructions

    def pack_dataset(
        self,
        obj: np.ndarray,
        key: str,
        file_format: str = "",
    ) -> tuple[Artifact, dict]:
        """
        Pack an array as a dataset.

        :param obj:         The aray to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is parquet.

        :return: The packed artifact and instructions.

        :raise MLRunInvalidArgumentError: IF the shape of the array is not 1D / 2D.
        """
        # Validate it's a 2D array:
        if len(obj.shape) > 2:
            raise MLRunInvalidArgumentError(
                f"Cannot log the given numpy array as a dataset. Only 2D arrays can be saved as dataset, but the array "
                f"is {len(obj.shape)}D (shape of {obj.shape}). Please specify to log it as a 'file' instead ('npy' "
                f"format) or as an 'object' (pickle)."
            )

        # Cast to a `pd.DataFrame`:
        data_frame = pd.DataFrame(data=obj)

        # Create the artifact:
        artifact = DatasetArtifact(key=key, df=data_frame, format=file_format)

        return artifact, {}

    def unpack_file(
        self,
        data_item: DataItem,
        file_format: Optional[str] = None,
        allow_pickle: bool = False,
    ) -> np.ndarray:
        """
        Unpack a numppy array from file.

        :param data_item:    The data item to unpack.
        :param file_format:  The file format to use for reading the array. Default is None - will be read by the file
                             extension.
        :param allow_pickle: Whether to allow loading pickled arrays in case of object type arrays. Only relevant to
                             'npy' format. Default is False for security reasons.

        :return: The unpacked array.
        """
        # Get the file:
        file_path = self.get_data_item_local_path(data_item=data_item)

        # Get the archive format by the file extension if needed:
        if file_format is None:
            file_format = NumPySupportedFormat.match_format(path=file_path)
        if (
            file_format is None
            or file_format in NumPySupportedFormat.get_multi_array_formats()
        ):
            raise MLRunInvalidArgumentError(
                f"File format of {data_item.key} ('{''.join(pathlib.Path(file_path).suffixes)}') is not supported. "
                f"Supported formats are: {' '.join(NumPySupportedFormat.get_single_array_formats())}"
            )

        # Read the object:
        formatter = NumPySupportedFormat.get_format_handler(fmt=file_format)
        load_kwargs = {}
        if file_format == NumPySupportedFormat.NPY:
            load_kwargs["allow_pickle"] = allow_pickle
        obj = formatter.load(file_path=file_path, **load_kwargs)

        return obj

    def unpack_dataset(self, data_item: DataItem) -> np.ndarray:
        """
        Unpack a numppy array from a dataset artifact.

        :param data_item: The data item to unpack.

        :return: The unpacked array.
        """
        # Get the artifact's data frame:
        data_frame = data_item.as_df()

        # Cast the data frame to a `np.ndarray` (1D arrays are returned as a 2D array with shape of 1xn, so we use
        # squeeze to decrease the redundant dimension):
        array = data_frame.to_numpy().squeeze()

        return array


class _NumPyNDArrayCollectionPackager(DefaultPackager):
    """
    A base packager for builtin python dictionaries and lists of numpy arrays as they share common artifact and file
    types.
    """

    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.FILE
    DEFAULT_UNPACKING_ARTIFACT_TYPE = ArtifactType.FILE
    PRIORITY = 4

    def pack_file(
        self,
        obj: NumPyArrayCollectionType,
        key: str,
        file_format: str = DEFAULT_NUMPPY_ARRAY_COLLECTION_FORMAT,
        **save_kwargs,
    ) -> tuple[Artifact, dict]:
        """
        Pack an array collection as a file by the given format.

        :param obj:         The aray collection to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is npy.
        :param save_kwargs: Additional keyword arguments to pass to the numpy save functions.

        :return: The packed artifact and instructions.
        """
        # Save to file:
        formatter = NumPySupportedFormat.get_format_handler(fmt=file_format)
        temp_directory = pathlib.Path(tempfile.mkdtemp())
        self.add_future_clearing_path(path=temp_directory)
        file_path = temp_directory / f"{key}.{file_format}"
        formatter.save(obj=obj, file_path=str(file_path), **save_kwargs)

        # Create the artifact and instructions (Note: only 'npz' format support saving object arrays and that will
        # require pickling, hence we set the required instruction):
        artifact = Artifact(key=key, src_path=os.path.abspath(file_path))
        instructions = {"file_format": file_format}
        if file_format == NumPySupportedFormat.NPZ and self._is_any_object_dtype(
            array_collection=obj
        ):
            instructions["allow_pickle"] = True

        return artifact, instructions

    def unpack_file(
        self,
        data_item: DataItem,
        file_format: Optional[str] = None,
        allow_pickle: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Unpack a numppy array collection from file.

        :param data_item:    The data item to unpack.
        :param file_format:  The file format to use for reading the array collection. Default is None - will be read by
                             the file extension.
        :param allow_pickle: Whether to allow loading pickled arrays in case of object type arrays. Only relevant to
                             'npz' format. Default is False for security reasons.

        :return: The unpacked array collection.
        """
        # Get the file:
        file_path = self.get_data_item_local_path(data_item=data_item)

        # Get the archive format by the file extension if needed:
        if file_format is None:
            file_format = NumPySupportedFormat.match_format(path=file_path)
        if (
            file_format is None
            or file_format in NumPySupportedFormat.get_single_array_formats()
        ):
            raise MLRunInvalidArgumentError(
                f"File format of {data_item.key} ('{''.join(pathlib.Path(file_path).suffixes)}') is not supported. "
                f"Supported formats are: {' '.join(NumPySupportedFormat.get_multi_array_formats())}"
            )

        # Read the object:
        formatter = NumPySupportedFormat.get_format_handler(fmt=file_format)
        load_kwargs = {}
        if file_format == NumPySupportedFormat.NPZ:
            load_kwargs["allow_pickle"] = allow_pickle
        obj = formatter.load(file_path=file_path, **load_kwargs)

        return obj

    @staticmethod
    def _is_any_object_dtype(
        array_collection: Union[np.ndarray, NumPyArrayCollectionType],
    ):
        """
        Check if any of the arrays in a collection is of type `object`.

        :param array_collection: The collection to check fo `object` dtype.

        :return: True if at least one array in the collection is an `object` array.
        """
        if isinstance(array_collection, list):
            return any(
                _NumPyNDArrayCollectionPackager._is_any_object_dtype(
                    array_collection=array
                )
                for array in array_collection
            )
        elif isinstance(array_collection, dict):
            return any(
                _NumPyNDArrayCollectionPackager._is_any_object_dtype(
                    array_collection=array
                )
                for array in array_collection.values()
            )
        return array_collection.dtype == np.object_


class NumPyNDArrayDictPackager(_NumPyNDArrayCollectionPackager):
    """
    ``dict[str, numpy.ndarray]`` packager.
    """

    PACKABLE_OBJECT_TYPE = dict[str, np.ndarray]

    def is_packable(
        self,
        obj: Any,
        artifact_type: Optional[str] = None,
        configurations: Optional[dict] = None,
    ) -> bool:
        """
        Check if the object provided is a dictionary of numpy arrays.

        :param obj:            The object to pack.
        :param artifact_type:  The artifact type to log the object as.
        :param configurations: The log hint configurations passed by the user.

        :return: True if packable and False otherwise.
        """
        # Check the obj is a dictionary with string keys and arrays as values:
        if not (
            isinstance(obj, dict)
            and all(
                isinstance(key, str) and isinstance(value, np.ndarray)
                for key, value in obj.items()
            )
        ):
            return False

        # Check the artifact type is supported:
        if artifact_type and artifact_type not in self.get_supported_artifact_types():
            return False

        # Check an edge case where the dictionary is empty (this packager will pack empty dictionaries only if given
        # specific file format, otherwise it will be packed by the `DictPackager`):
        if not obj:
            return (
                configurations.get("file_format", None)
                in NumPySupportedFormat().get_multi_array_formats()
            )

        return True

    def pack_result(self, obj: dict[str, np.ndarray], key: str) -> dict:
        """
        Pack a dictionary of numpy arrays as a result.

        :param obj: The arrays dictionary to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        return {
            key: {
                array_key: array_value.tolist()
                for array_key, array_value in obj.items()
            }
        }

    def unpack_file(
        self,
        data_item: DataItem,
        file_format: Optional[str] = None,
        allow_pickle: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Unpack a numppy array dictionary from file.

        :param data_item:    The data item to unpack.
        :param file_format:  The file format to use for reading the arrays dictionary. Default is None - will be read by
                             the file extension.
        :param allow_pickle: Whether to allow loading pickled arrays in case of object type arrays. Only relevant to
                             'npz' format. Default is False for security reasons.

        :return: The unpacked array.
        """
        # Load the object:
        obj = super().unpack_file(
            data_item=data_item, file_format=file_format, allow_pickle=allow_pickle
        )

        # The returned object is a mapping of type NpzFile, so we cast it to a dictionary:
        return {key: array for key, array in obj.items()}


class NumPyNDArrayListPackager(_NumPyNDArrayCollectionPackager):
    """
    ``list[numpy.ndarray]`` packager.
    """

    PACKABLE_OBJECT_TYPE = list[np.ndarray]

    def is_packable(
        self,
        obj: Any,
        artifact_type: Optional[str] = None,
        configurations: Optional[dict] = None,
    ) -> bool:
        """
        Check if the object provided is a list of numpy arrays.

        :param obj:            The object to pack.
        :param artifact_type:  The artifact type to log the object as.
        :param configurations: The log hint configurations passed by the user.

        :return: True if packable and False otherwise.
        """
        # Check the obj is a list with arrays as values:
        if not (
            isinstance(obj, list)
            and all(isinstance(value, np.ndarray) for value in obj)
        ):
            return False

        # Check the artifact type is supported:
        if artifact_type and artifact_type not in self.get_supported_artifact_types():
            return False

        # Check an edge case where the list is empty (this packager will pack empty lists only if given specific file
        # format, otherwise it will be packed by the `ListPackager`):
        if not obj:
            return (
                configurations.get("file_format", None)
                in NumPySupportedFormat().get_multi_array_formats()
            )

        return True

    def pack_result(self, obj: list[np.ndarray], key: str) -> dict:
        """
        Pack a list of numpy arrays as a result.

        :param obj: The arrays list to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        return {key: [array.tolist() for array in obj]}

    def unpack_file(
        self,
        data_item: DataItem,
        file_format: Optional[str] = None,
        allow_pickle: bool = False,
    ) -> list[np.ndarray]:
        """
        Unpack a numppy array list from file.

        :param data_item:    The data item to unpack.
        :param file_format:  The file format to use for reading the arrays list. Default is None - will be read by the
                             file extension.
        :param allow_pickle: Whether to allow loading pickled arrays in case of object type arrays. Only relevant to
                             'npz' format. Default is False for security reasons.

        :return: The unpacked array.
        """
        # Load the object:
        obj = super().unpack_file(
            data_item=data_item, file_format=file_format, allow_pickle=allow_pickle
        )

        # The returned object is a mapping of type NpzFile, so we cast it to a list:
        return list(obj.values())


class NumPyNumberPackager(DefaultPackager):
    """
    ``numpy.number`` packager. It is also used for all `number` inheriting numpy objects (`float32`, uint8, etc.).
    """

    PACKABLE_OBJECT_TYPE = np.number
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.RESULT
    PACK_SUBCLASSES = True  # To include all dtypes ('float32', 'uint8', ...)

    def pack_result(self, obj: np.number, key: str) -> dict:
        """
        Pack a numpy number as a result.

        :param obj: The number to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        return super().pack_result(obj=obj.item(), key=key)
