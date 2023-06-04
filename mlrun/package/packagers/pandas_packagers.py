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
from typing import Tuple, Union

import pandas as pd

from mlrun.artifacts import Artifact, DatasetArtifact
from mlrun.datastore import DataItem
from mlrun.errors import MLRunInvalidArgumentError

from ..utils import ArtifactType, SupportedFormat
from .default_packager import DefaultPackager

# A type hint for all pandas data objects available for saving to and reading from files:
PandasDataType = Union[pd.DataFrame, pd.Series]


class _Formatter(ABC):
    """
    An abstract class for a pandas formatter - supporting saving and loading dataframes to and from specific file type.
    """

    @classmethod
    @abstractmethod
    def to(cls, obj: PandasDataType, file_path: str, **to_kwargs: dict):
        """
        Save the given dataframe / series to the file path given.

        :param obj:       The dataframe / series to save.
        :param file_path: The file to save to.
        :param to_kwargs: Additional keyword arguments to pass to the relevant `to_x` function.
        """
        pass

    @classmethod
    @abstractmethod
    def read(cls, file_path: str, **read_kwargs: dict) -> PandasDataType:
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
    def read(cls, file_path: str, **read_kwargs: dict) -> PandasDataType:
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
        obj.to_csv(path=file_path, **to_kwargs)

    @classmethod
    def read(cls, file_path: str, **read_kwargs: dict) -> PandasDataType:
        """
        Read the dataframe / series from the given parquet file path.

        :param file_path:   The file to read the dataframe from.
        :param read_kwargs: Additional keyword arguments to pass to the `read_csv` function.

        :return: The loaded dataframe / series.
        """
        return pd.read_csv(path=file_path, **read_kwargs)


class PandasSupportedFormat(SupportedFormat[_Formatter]):
    """
    Library of Pandas formats (file extensions) supported by the Pandas packagers.
    """

    PARQUET = "parquet"
    CSV = "csv"
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


class PandasSeriesPackager(DefaultPackager):
    """
    ``pd.Series`` packager.
    """

    PACKABLE_OBJECT_TYPE = pd.Series
    DEFAULT_PACKING_ARTIFACT_TYPE = ArtifactType.FILE

    @classmethod
    def pack_file(
        cls,
        obj: pd.Series,
        key: str,
        file_format: str = DEFAULT_PANDAS_FORMAT,
        **to_kwargs,
    ) -> Tuple[Artifact, dict]:
        """
        Pack a series as a file by the given format.

        :param obj:         The series to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is parquet.
        :param to_kwargs:   Additional keyword arguments to pass to the pandas `to_x` functions.

        :return: The packed artifact and instructions.
        """
        # Save to file:
        formatter = PandasSupportedFormat.get_format_handler(fmt=file_format)
        temp_directory = pathlib.Path(tempfile.mkdtemp())
        cls.add_future_clearing_path(path=temp_directory)
        file_path = temp_directory / f"{key}.{file_format}"
        formatter.to(obj=obj, file_path=str(file_path), **to_kwargs)

        # Create the artifact and instructions:
        artifact = Artifact(key=key, src_path=os.path.abspath(file_path))

        return artifact, {"file_format": file_format}

    @classmethod
    def unpack_file(cls, data_item: DataItem, file_format: str = None) -> pd.Series:
        """
        Unpack a pandas series from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the series. Default is None - will be read by the file
                            extension.

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

        return obj


class PandasDataFramePackager(PandasSeriesPackager):
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
        is_artifact = data_item.is_artifact()
        if is_artifact and is_artifact == "datasets":
            return ArtifactType.DATASET
        return ArtifactType.FILE

    @classmethod
    def pack_dataset(cls, obj: pd.DataFrame, key: str, file_format: str = "parquet"):
        """
        Pack a pandas dataframe as a dataset.

        :param obj:         The dataframe to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is parquet.

        :return: The packed artifact and instructions.
        """
        return DatasetArtifact(key=key, df=obj, format=file_format), None

    @classmethod
    def unpack_dataset(cls, data_item: DataItem):
        """
        Unpack a padnas dataframe from a dataset artifact.

        :param data_item: The data item to unpack.

        :return: The unpacked dataframe.
        """
        return data_item.as_df()
