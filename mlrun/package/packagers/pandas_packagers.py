# Copyright 2018 Iguazio
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
from typing import Any, List, Tuple, Union

import pandas as pd

from mlrun.artifacts import Artifact, DatasetArtifact
from mlrun.datastore import DataItem
from mlrun.errors import MLRunInvalidArgumentError

from ..utils import ArtifactType, SupportedFormat
from .default_packager import DefaultPackager


class _Formatter(ABC):
    """
    An abstract class for a pandas formatter - supporting saving and loading dataframes to and from specific file type.
    """

    @classmethod
    @abstractmethod
    def to(cls, obj: pd.DataFrame, file_path: str, **to_kwargs: dict):
        """
        Save the given dataframe / series to the file path given.

        :param obj:       The dataframe / series to save.
        :param file_path: The file to save to.
        :param to_kwargs: Additional keyword arguments to pass to the relevant `to_x` function.
        """
        pass

    @classmethod
    @abstractmethod
    def read(cls, file_path: str, **read_kwargs: dict) -> pd.DataFrame:
        """
        Read the dataframe / series from the given file path.

        :param file_path:   The file to read the dataframe from.
        :param read_kwargs: Additional keyword arguments to pass to the relevant read function of pandas.

        :return: The loaded dataframe / series.
        """
        pass


class _ParquetFormatter(_Formatter):
    """
    A static class for managing pandas parquet files.
    """

    @classmethod
    def to(cls, obj: pd.DataFrame, file_path: str, **to_kwargs: dict):
        """
        Save the given dataframe / series to the file path given.

        :param obj:       The dataframe / series to save.
        :param file_path: The file to save to.
        :param to_kwargs: Additional keyword arguments to pass to the relevant `to_parquet` function.
        """
        obj.to_parquet(path=file_path, **to_kwargs)

    @classmethod
    def read(cls, file_path: str, **read_kwargs: dict) -> pd.DataFrame:
        """
        Read the dataframe / series from the given parquet file path.

        :param file_path:   The file to read the dataframe from.
        :param read_kwargs: Additional keyword arguments to pass to the `read_parquet` function.

        :return: The loaded dataframe / series.
        """
        return pd.read_parquet(path=file_path, **read_kwargs)


class _CSVFormatter(_Formatter):
    """
    A static class for managing pandas csv files.
    """

    @classmethod
    def to(cls, obj: pd.DataFrame, file_path: str, **to_kwargs: dict):
        """
        Save the given dataframe / series to the file path given.

        :param obj:       The dataframe / series to save.
        :param file_path: The file to save to.
        :param to_kwargs: Additional keyword arguments to pass to the relevant `to_csv` function.
        """
        obj.to_csv(path_or_buf=file_path, **to_kwargs)

    @classmethod
    def read(cls, file_path: str, **read_kwargs: dict) -> pd.DataFrame:
        """
        Read the dataframe / series from the given parquet file path.

        :param file_path:   The file to read the dataframe from.
        :param read_kwargs: Additional keyword arguments to pass to the `read_csv` function.

        :return: The loaded dataframe / series.
        """
        return pd.read_csv(filepath_or_buffer=file_path, **read_kwargs)


class PandasSupportedFormat(SupportedFormat[_Formatter]):
    """
    Library of Pandas formats (file extensions) supported by the Pandas packagers.
    """

    PARQUET = "parquet"
    CSV = "csv"
    # TODO: Add support for all the below formats:
    # H5 = "h5"
    # XML = "xml"
    # XLSX = "xlsx"
    # HTML = "html"
    # JSON = "json"
    # FEATHER = "feather"
    # ORC = "orc"

    _FORMAT_HANDLERS_MAP = {
        PARQUET: _ParquetFormatter,
        CSV: _CSVFormatter,
        # H5: _H5Formatter,
        # XML: _XMLFormatter,
        # XLSX: _XLSXFormatter,
        # HTML: _HTMLFormatter,
        # JSON: _JSONFormatter,
        # FEATHER: _FeatherFormatter,
        # ORC: _ORCFormatter,
    }


# Default file formats for pandas DataFrame and Series file artifacts:
DEFAULT_PANDAS_FORMAT = PandasSupportedFormat.PARQUET
NON_STRING_COLUMN_NAMES_DEFAULT_PANDAS_FORMAT = PandasSupportedFormat.CSV


class PandasDataFramePackager(DefaultPackager):
    """
    ``pd.DataFrame`` packager.
    """

    PACKABLE_OBJECT_TYPE = pd.DataFrame
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.DATASET

    @classmethod
    def get_default_unpacking_artifact_type(cls, data_item: DataItem) -> str:
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

    @classmethod
    def pack_result(cls, obj: pd.DataFrame, key: str) -> dict:
        """
        Pack a dataframe as a result.

        :param obj: The dataframe to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        # Parse to dictionary according to the indexes in the dataframe:
        if len(obj.index.names) > 1:
            # Multiple indexes:
            orient = "split"
        elif obj.index.name is not None:
            # Not a default index (user would likely want to keep it):
            orient = "dict"
        else:
            # Default index can be ignored:
            orient = "list"

        # Cast to dictionary:
        dataframe_dictionary = obj.to_dict(orient=orient)

        # Prepare the result (casting tuples to lists):
        dataframe_dictionary = PandasDataFramePackager._prepare_result(
            obj=dataframe_dictionary
        )

        return super().pack_result(obj=dataframe_dictionary, key=key)

    @classmethod
    def pack_file(
        cls,
        obj: pd.DataFrame,
        key: str,
        file_format: str = None,
        **to_kwargs,
    ) -> Tuple[Artifact, dict]:
        """
        Pack a dataframe as a file by the given format.

        :param obj:         The series to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is parquet or csv (depends on the column names as
                            parquet cannot be used for non string column names).
        :param to_kwargs:   Additional keyword arguments to pass to the pandas `to_x` functions.

        :return: The packed artifact and instructions.
        """
        # Set default file format if not given:
        if file_format is None:
            file_format = (
                DEFAULT_PANDAS_FORMAT
                if all(isinstance(name, str) for name in obj.columns)
                else NON_STRING_COLUMN_NAMES_DEFAULT_PANDAS_FORMAT
            )

        # Get the indexes as they may get changed during saving in some file formats:
        indexes_names = list(obj.index.names)  # No index will yield '[None]'.

        # Save to file:
        formatter = PandasSupportedFormat.get_format_handler(fmt=file_format)
        temp_directory = pathlib.Path(tempfile.mkdtemp())
        cls.add_future_clearing_path(path=temp_directory)
        file_path = temp_directory / f"{key}.{file_format}"
        formatter.to(obj=obj, file_path=str(file_path), **to_kwargs)

        # Create the artifact and instructions:
        artifact = Artifact(key=key, src_path=os.path.abspath(file_path))

        return artifact, {"file_format": file_format, "indexes_names": indexes_names}

    @classmethod
    def pack_dataset(cls, obj: pd.DataFrame, key: str, file_format: str = "parquet"):
        """
        Pack a pandas dataframe as a dataset.

        :param obj:         The dataframe to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is parquet.

        :return: The packed artifact and instructions.
        """
        return DatasetArtifact(key=key, df=obj, format=file_format), {}

    @classmethod
    def unpack_file(
        cls,
        data_item: DataItem,
        file_format: str = None,
        indexes_names: List[Union[str, int]] = None,
    ) -> pd.DataFrame:
        """
        Unpack a pandas dataframe from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the series. Default is None - will be read by the file
                            extension.
        :param indexes_names: Names of the indexes in the dataframe.

        :return: The unpacked series.
        """
        # Get the file:
        file_path = data_item.local()
        cls.add_future_clearing_path(path=file_path)

        # Get the archive format by the file extension if needed:
        if file_format is None:
            file_format = PandasSupportedFormat.match_format(path=file_path)
        if file_format is None:
            raise MLRunInvalidArgumentError(
                f"File format of {data_item.key} ('{''.join(pathlib.Path(file_path).suffixes)}') is not supported. "
                f"Supported formats are: {' '.join(PandasSupportedFormat.get_all_formats())}"
            )

        # Read the object:
        formatter = PandasSupportedFormat.get_format_handler(fmt=file_format)
        obj = formatter.read(file_path=file_path)

        # Set indexes if given by instructions and the default index (without name) is currently set in the dataframe:
        if indexes_names is not None and list(obj.index.names) == [None]:
            if indexes_names == [None]:
                # If the default index was used (an index without a column name), it will be the first column, and it's
                # name may be 'Unnamed: 0' so we need override it:
                if obj.columns[0] == "Unnamed: 0":
                    obj.set_index(keys=["Unnamed: 0"], drop=True, inplace=True)
                obj.index.set_names(names=[None], inplace=True)
            else:
                # Otherwise, simply set the original indexes from the available columns:
                obj.set_index(keys=indexes_names, drop=True, inplace=True)

        return obj

    @classmethod
    def unpack_dataset(cls, data_item: DataItem):
        """
        Unpack a padnas dataframe from a dataset artifact.

        :param data_item: The data item to unpack.

        :return: The unpacked dataframe.
        """
        return data_item.as_df()

    @staticmethod
    def _prepare_result(obj: Union[list, dict, tuple]) -> Any:
        """
        A dataframe can be logged as a result when it being cast to a dictionary. If the dataframe has multiple indexes,
        pandas store them as a tuple, which is not json serializable, so we cast them into lists.

        :param obj: The dataframe dictionary (or list and tuple as it is recursive).

        :return: Prepared result.
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[
                    PandasDataFramePackager._prepare_result(obj=key)
                ] = PandasDataFramePackager._prepare_result(obj=value)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                obj[i] = PandasDataFramePackager._prepare_result(obj=value)
        elif isinstance(obj, tuple):
            obj = [PandasDataFramePackager._prepare_result(obj=value) for value in obj]
        return obj


class PandasSeriesPackager(PandasDataFramePackager):
    """
    ``pd.Series`` packager.
    """

    PACKABLE_OBJECT_TYPE = pd.Series
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.FILE

    @classmethod
    def get_supported_artifact_types(cls) -> List[str]:
        """
        Get all the supported artifact types on this packager. It will be the same as `PandasDataFramePackager` but
        without the 'dataset' artifact type support.

        :return: A list of all the supported artifact types.
        """
        supported_artifacts = super().get_supported_artifact_types()
        supported_artifacts.remove("dataset")
        return supported_artifacts

    @classmethod
    def pack_result(cls, obj: pd.Series, key: str) -> dict:
        """
        Pack a series as a result.

        :param obj: The series to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        return super().pack_result(obj=pd.DataFrame(obj), key=key)

    @classmethod
    def pack_file(
        cls,
        obj: pd.Series,
        key: str,
        file_format: str = None,
        **to_kwargs,
    ) -> Tuple[Artifact, dict]:
        """
        Pack a series as a file by the given format.

        :param obj:         The series to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is parquet or csv (depends on the column names as
                            parquet cannot be used for non string column names).
        :param to_kwargs:   Additional keyword arguments to pass to the pandas `to_x` functions.

        :return: The packed artifact and instructions.
        """
        # Get the series column name:
        column_name = obj.name

        # Cast to dataframe and call the parent `pack_file`:
        artifact, instructions = super().pack_file(
            obj=pd.DataFrame(obj), key=key, file_format=file_format, **to_kwargs
        )

        # Return the artifact with the updated instructions:
        return artifact, {**instructions, "column_name": column_name}

    @classmethod
    def unpack_file(
        cls,
        data_item: DataItem,
        file_format: str = None,
        indexes_names: List[Union[str, int]] = None,
        column_name: Union[str, int] = None,
    ) -> pd.Series:
        """
        Unpack a pandas series from file.

        :param data_item:     The data item to unpack.
        :param file_format:   The file format to use for reading the series. Default is None - will be read by the file
                              extension.
        :param indexes_names: Names of the indexes in the series.
        :param column_name:   The name of the series column.

        :return: The unpacked series.
        """
        # Read the object:
        obj = super().unpack_file(
            data_item=data_item, file_format=file_format, indexes_names=indexes_names
        )

        # Cast the dataframe into a series:
        if len(obj.columns) != 1:
            raise MLRunInvalidArgumentError(
                f"The data item received is of a `pandas.DataFrame` with more than one column: "
                f"{', '.join(obj.columns)}. Hence it cannot be turned into a `pandas.Series`."
            )
        obj = obj[obj.columns[0]]

        # Edit the column name:
        if column_name is not None:
            obj.name = column_name

        return obj
