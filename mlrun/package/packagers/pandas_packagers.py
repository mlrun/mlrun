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
import importlib
import os
import pathlib
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

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
    def to(
        cls, obj: pd.DataFrame, file_path: str, flatten: bool = True, **to_kwargs
    ) -> dict:
        """
        Save the given dataframe to the file path given.

        :param obj:       The dataframe to save.
        :param file_path: The file to save to.
        :param flatten:   Whether to flatten the dataframe before saving. For some formats it is mandatory to enable
                          flattening, otherwise saving and loading the dataframe will cause unexpected behavior
                          especially in case it is multi-level or multi-index. Default to True.
        :param to_kwargs: Additional keyword arguments to pass to the relevant `to_x` function.

        :return A dictionary of keyword arguments for reading the dataframe from file.
        """
        pass

    @classmethod
    @abstractmethod
    def read(
        cls, file_path: str, unflatten_kwargs: Optional[dict] = None, **read_kwargs
    ) -> pd.DataFrame:
        """
        Read the dataframe from the given file path.

        :param file_path:        The file to read the dataframe from.
        :param unflatten_kwargs: Unflatten keyword arguments for unflattening the read dataframe.
        :param read_kwargs:      Additional keyword arguments to pass to the relevant read function of pandas.

        :return: The loaded dataframe.
        """
        pass

    @staticmethod
    def _flatten_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Flatten the dataframe: moving all indexes to be columns at the start (from column 0) and lowering the columns
        levels to 1, renaming them from tuples. All columns and index info is stored so it can be unflatten later on.

        :param dataframe: The dataframe to flatten.

        :return: The flat dataframe.
        """
        # Save columns info:
        columns = list(dataframe.columns)
        if isinstance(dataframe.columns, pd.MultiIndex):
            columns = [list(column_tuple) for column_tuple in columns]
        columns_levels = list(dataframe.columns.names)

        # Save index info:
        index_levels = list(dataframe.index.names)

        # Turn multi-index columns into single columns:
        if len(columns_levels) > 1:
            # We turn the column tuple into a string to eliminate parsing issues during savings to text formats:
            dataframe.columns = pd.Index(
                "-".join(column_tuple) for column_tuple in columns
            )

        # Rename indexes in case they appear in the columns so it won't get overriden when the index reset:
        dataframe.index.set_names(
            names=[
                name
                if name is not None and name not in dataframe.columns
                else f"INDEX_{name}_{i}"
                for i, name in enumerate(dataframe.index.names)
            ],
            inplace=True,
        )

        # Reset the index, moving the current index to a column:
        dataframe.reset_index(inplace=True)

        return dataframe, {
            "columns": columns,
            "columns_levels": columns_levels,
            "index_levels": index_levels,
        }

    @staticmethod
    def _unflatten_dataframe(
        dataframe: pd.DataFrame,
        columns: list,
        columns_levels: list,
        index_levels: list,
    ) -> pd.DataFrame:
        """
        Unflatten the dataframe, moving the indexes from the columns and resuming the columns levels and names.

        :param dataframe:      The dataframe to unflatten.
        :param columns:        The original list of columns.
        :param columns_levels: The original columns levels names.
        :param index_levels:   The original index levels names.

        :return: The un-flatted dataframe.
        """
        # Move back index from columns:
        dataframe.set_index(
            keys=list(dataframe.columns[: len(index_levels)]), inplace=True
        )
        dataframe.index.set_names(names=index_levels, inplace=True)

        # Set the columns back in case they were multi-leveled:
        if len(columns_levels) > 1:
            dataframe.columns = pd.MultiIndex.from_tuples(
                tuples=columns, names=columns_levels
            )
        else:
            dataframe.columns.set_names(names=columns_levels, inplace=True)

        return dataframe


class _ParquetFormatter(_Formatter):
    """
    A static class for managing pandas parquet files.
    """

    @classmethod
    def to(
        cls, obj: pd.DataFrame, file_path: str, flatten: bool = True, **to_kwargs
    ) -> dict:
        """
        Save the given dataframe to the parquet file path given.

        :param obj:       The dataframe to save.
        :param file_path: The file to save to.
        :param flatten:   Ignored for parquet format.
        :param to_kwargs: Additional keyword arguments to pass to the `to_parquet` function.

        :return A dictionary of keyword arguments for reading the dataframe from file.
        """
        obj.to_parquet(path=file_path, **to_kwargs)
        return {}

    @classmethod
    def read(
        cls, file_path: str, unflatten_kwargs: Optional[dict] = None, **read_kwargs
    ) -> pd.DataFrame:
        """
        Read the dataframe from the given parquet file path.

        :param file_path:        The file to read the dataframe from.
        :param unflatten_kwargs: Ignored for parquet format.
        :param read_kwargs:      Additional keyword arguments to pass to the `read_parquet` function.

        :return: The loaded dataframe.
        """
        return pd.read_parquet(path=file_path, **read_kwargs)


class _CSVFormatter(_Formatter):
    """
    A static class for managing pandas csv files.
    """

    @classmethod
    def to(
        cls, obj: pd.DataFrame, file_path: str, flatten: bool = True, **to_kwargs
    ) -> dict:
        """
        Save the given dataframe to the csv file path given.

        :param obj:       The dataframe to save.
        :param file_path: The file to save to.
        :param flatten:   Whether to flatten the dataframe before saving. For some formats it is mandatory to enable
                          flattening, otherwise saving and loading the dataframe will cause unexpected behavior
                          especially in case it is multi-level or multi-index. Default to True.
        :param to_kwargs: Additional keyword arguments to pass to the `to_csv` function.

        :return A dictionary of keyword arguments for reading the dataframe from file.
        """
        # Flatten the dataframe (this format have problems saving multi-level dataframes):
        instructions = {}
        if flatten:
            obj, unflatten_kwargs = cls._flatten_dataframe(dataframe=obj)
            instructions["unflatten_kwargs"] = unflatten_kwargs

        # Write to csv:
        obj.to_csv(path_or_buf=file_path, **to_kwargs)

        return instructions

    @classmethod
    def read(
        cls, file_path: str, unflatten_kwargs: Optional[dict] = None, **read_kwargs
    ) -> pd.DataFrame:
        """
        Read the dataframe from the given csv file path.

        :param file_path:        The file to read the dataframe from.
        :param unflatten_kwargs: Unflatten keyword arguments for unflattening the read dataframe.
        :param read_kwargs:      Additional keyword arguments to pass to the `read_csv` function.

        :return: The loaded dataframe.
        """
        # Read the csv:
        obj = pd.read_csv(filepath_or_buffer=file_path, **read_kwargs)

        # Check if it was flattened in packing:
        if unflatten_kwargs is not None:
            # Remove the default index (joined with reset index):
            if obj.columns[0] == "Unnamed: 0":
                obj.drop(columns=["Unnamed: 0"], inplace=True)
            # Unflatten the dataframe:
            obj = cls._unflatten_dataframe(dataframe=obj, **unflatten_kwargs)

        return obj


class _H5Formatter(_Formatter):
    """
    A static class for managing pandas h5 files.
    """

    @classmethod
    def to(
        cls, obj: pd.DataFrame, file_path: str, flatten: bool = True, **to_kwargs
    ) -> dict:
        """
        Save the given dataframe to the h5 file path given.

        :param obj:       The dataframe to save.
        :param file_path: The file to save to.
        :param flatten:   Ignored for h5 format.
        :param to_kwargs: Additional keyword arguments to pass to the `to_hdf` function.

        :return A dictionary of keyword arguments for reading the dataframe from file.
        """
        # If user didn't provide a key for the dataframe, use default key 'table':
        key = to_kwargs.pop("key", "table")

        # Write to h5:
        obj.to_hdf(path_or_buf=file_path, key=key, **to_kwargs)

        return {"key": key}

    @classmethod
    def read(
        cls, file_path: str, unflatten_kwargs: Optional[dict] = None, **read_kwargs
    ) -> pd.DataFrame:
        """
        Read the dataframe from the given h5 file path.

        :param file_path:        The file to read the dataframe from.
        :param unflatten_kwargs: Ignored for h5 format.
        :param read_kwargs:      Additional keyword arguments to pass to the `read_hdf` function.

        :return: The loaded dataframe.
        """
        return pd.read_hdf(path_or_buf=file_path, **read_kwargs)


class _XMLFormatter(_Formatter):
    """
    A static class for managing pandas xml files.
    """

    @classmethod
    def to(
        cls, obj: pd.DataFrame, file_path: str, flatten: bool = True, **to_kwargs
    ) -> dict:
        """
        Save the given dataframe to the xml file path given.

        :param obj:       The dataframe to save.
        :param file_path: The file to save to.
        :param flatten:   Whether to flatten the dataframe before saving. For some formats it is mandatory to enable
                          flattening, otherwise saving and loading the dataframe will cause unexpected behavior
                          especially in case it is multi-level or multi-index. Default to True.
        :param to_kwargs: Additional keyword arguments to pass to the `to_xml` function.

        :return A dictionary of keyword arguments for reading the dataframe from file.
        """
        # Get the parser (if not provided, try to use `lxml`, otherwise `etree`):
        parser = to_kwargs.pop("parser", None)
        if parser is None:
            try:
                importlib.import_module("lxml")
                parser = "lxml"
            except ModuleNotFoundError:
                parser = "etree"
        instructions = {"parser": parser}

        # Flatten the dataframe (this format have problems saving multi-level dataframes):
        if flatten:
            obj, unflatten_kwargs = cls._flatten_dataframe(dataframe=obj)
            instructions["unflatten_kwargs"] = unflatten_kwargs

        # Write to xml:
        obj.to_xml(path_or_buffer=file_path, parser="etree", **to_kwargs)

        return instructions

    @classmethod
    def read(
        cls, file_path: str, unflatten_kwargs: Optional[dict] = None, **read_kwargs
    ) -> pd.DataFrame:
        """
        Read the dataframe from the given xml file path.

        :param file_path:        The file to read the dataframe from.
        :param unflatten_kwargs: Unflatten keyword arguments for unflattening the read dataframe.
        :param read_kwargs:      Additional keyword arguments to pass to the `read_xml` function.

        :return: The loaded dataframe.
        """
        # Read the xml:
        obj = pd.read_xml(path_or_buffer=file_path, **read_kwargs)

        # Check if it was flattened in packing:
        if unflatten_kwargs is not None:
            # Remove the default index (joined with reset index):
            if obj.columns[0] == "index":
                obj.drop(columns=["index"], inplace=True)
            # Unflatten the dataframe:
            obj = cls._unflatten_dataframe(dataframe=obj, **unflatten_kwargs)

        return obj


class _XLSXFormatter(_Formatter):
    """
    A static class for managing pandas xlsx files.
    """

    @classmethod
    def to(
        cls, obj: pd.DataFrame, file_path: str, flatten: bool = True, **to_kwargs
    ) -> dict:
        """
        Save the given dataframe to the xlsx file path given.

        :param obj:       The dataframe to save.
        :param file_path: The file to save to.
        :param flatten:   Whether to flatten the dataframe before saving. For some formats it is mandatory to enable
                          flattening, otherwise saving and loading the dataframe will cause unexpected behavior
                          especially in case it is multi-level or multi-index. Default to True.
        :param to_kwargs: Additional keyword arguments to pass to the `to_excel` function.
        """
        # Get the engine to pass when unpacked:
        instructions = {"engine": to_kwargs.get("engine", None)}

        # Flatten the dataframe (this format have problems saving multi-level dataframes):
        if flatten:
            obj, unflatten_kwargs = cls._flatten_dataframe(dataframe=obj)
            instructions["unflatten_kwargs"] = unflatten_kwargs

        # Write to xlsx:
        obj.to_excel(excel_writer=file_path, **to_kwargs)

        return instructions

    @classmethod
    def read(
        cls, file_path: str, unflatten_kwargs: Optional[dict] = None, **read_kwargs
    ) -> pd.DataFrame:
        """
        Read the dataframe from the given xlsx file path.

        :param file_path:        The file to read the dataframe from.
        :param unflatten_kwargs: Unflatten keyword arguments for unflattening the read dataframe.
        :param read_kwargs:      Additional keyword arguments to pass to the `read_excel` function.

        :return: The loaded dataframe.
        """
        # Read the xlsx:
        obj = pd.read_excel(io=file_path, **read_kwargs)

        # Check if it was flattened in packing:
        if unflatten_kwargs is not None:
            # Remove the default index (joined with reset index):
            if obj.columns[0] == "Unnamed: 0":
                obj.drop(columns=["Unnamed: 0"], inplace=True)
            # Unflatten the dataframe:
            obj = cls._unflatten_dataframe(dataframe=obj, **unflatten_kwargs)

        return obj


class _HTMLFormatter(_Formatter):
    """
    A static class for managing pandas html files.
    """

    @classmethod
    def to(
        cls, obj: pd.DataFrame, file_path: str, flatten: bool = True, **to_kwargs
    ) -> dict:
        """
        Save the given dataframe to the html file path given.

        :param obj:       The dataframe to save.
        :param file_path: The file to save to.
        :param flatten:   Whether to flatten the dataframe before saving. For some formats it is mandatory to enable
                          flattening, otherwise saving and loading the dataframe will cause unexpected behavior
                          especially in case it is multi-level or multi-index. Default to True.
        :param to_kwargs: Additional keyword arguments to pass to the `to_html` function.

        :return A dictionary of keyword arguments for reading the dataframe from file.
        """
        # Flatten the dataframe (this format have problems saving multi-level dataframes):
        instructions = {}
        if flatten:
            obj, unflatten_kwargs = cls._flatten_dataframe(dataframe=obj)
            instructions["unflatten_kwargs"] = unflatten_kwargs

        # Write to html:
        obj.to_html(buf=file_path, **to_kwargs)
        return instructions

    @classmethod
    def read(
        cls, file_path: str, unflatten_kwargs: Optional[dict] = None, **read_kwargs
    ) -> pd.DataFrame:
        """
        Read dataframes from the given html file path.

        :param file_path:        The file to read the dataframe from.
        :param unflatten_kwargs: Unflatten keyword arguments for unflattening the read dataframe.
        :param read_kwargs:      Additional keyword arguments to pass to the `read_html` function.

        :return: The loaded dataframe.
        """
        # Read the html:
        obj = pd.read_html(io=file_path, **read_kwargs)[0]

        # Check if it was flattened in packing:
        if unflatten_kwargs is not None:
            # Remove the default index (joined with reset index):
            if obj.columns[0] == "Unnamed: 0":
                obj.drop(columns=["Unnamed: 0"], inplace=True)
            # Unflatten the dataframe:
            obj = cls._unflatten_dataframe(dataframe=obj, **unflatten_kwargs)

        return obj


class _JSONFormatter(_Formatter):
    """
    A static class for managing pandas json files.
    """

    @classmethod
    def to(
        cls, obj: pd.DataFrame, file_path: str, flatten: bool = True, **to_kwargs
    ) -> dict:
        """
        Save the given dataframe to the json file path given.

        :param obj:       The dataframe to save.
        :param file_path: The file to save to.
        :param flatten:   Whether to flatten the dataframe before saving. For some formats it is mandatory to enable
                          flattening, otherwise saving and loading the dataframe will cause unexpected behavior
                          especially in case it is multi-level or multi-index. Default to True.
        :param to_kwargs: Additional keyword arguments to pass to the `to_json` function.

        :return A dictionary of keyword arguments for reading the dataframe from file.
        """
        # Get the orient to pass when unpacked:
        instructions = {"orient": to_kwargs.get("orient", None)}

        # Flatten the dataframe (this format have problems saving multi-level dataframes):
        if flatten:
            obj, unflatten_kwargs = cls._flatten_dataframe(dataframe=obj)
            instructions["unflatten_kwargs"] = unflatten_kwargs

        # Write to json:
        obj.to_json(path_or_buf=file_path, **to_kwargs)

        return instructions

    @classmethod
    def read(
        cls, file_path: str, unflatten_kwargs: Optional[dict] = None, **read_kwargs
    ) -> pd.DataFrame:
        """
        Read dataframes from the given json file path.

        :param file_path:        The file to read the dataframe from.
        :param unflatten_kwargs: Unflatten keyword arguments for unflattening the read dataframe.
        :param read_kwargs:      Additional keyword arguments to pass to the `read_json` function.

        :return: The loaded dataframe.
        """
        # Read the json:
        obj = pd.read_json(path_or_buf=file_path, **read_kwargs)

        # Check if it was flattened in packing:
        if unflatten_kwargs is not None:
            obj = cls._unflatten_dataframe(dataframe=obj, **unflatten_kwargs)

        return obj


class _FeatherFormatter(_Formatter):
    """
    A static class for managing pandas feather files.
    """

    @classmethod
    def to(
        cls, obj: pd.DataFrame, file_path: str, flatten: bool = True, **to_kwargs
    ) -> dict:
        """
        Save the given dataframe to the feather file path given.

        :param obj:       The dataframe to save.
        :param file_path: The file to save to.
        :param flatten:   Whether to flatten the dataframe before saving. For some formats it is mandatory to enable
                          flattening, otherwise saving and loading the dataframe will cause unexpected behavior
                          especially in case it is multi-level or multi-index. Default to True.
        :param to_kwargs: Additional keyword arguments to pass to the `to_feather` function.

        :return A dictionary of keyword arguments for reading the dataframe from file.
        """
        # Flatten the dataframe (this format have problems saving multi-level dataframes):
        instructions = {}
        if flatten:
            obj, unflatten_kwargs = cls._flatten_dataframe(dataframe=obj)
            instructions["unflatten_kwargs"] = unflatten_kwargs

        # Write to feather:
        obj.to_feather(path=file_path, **to_kwargs)

        return instructions

    @classmethod
    def read(
        cls, file_path: str, unflatten_kwargs: Optional[dict] = None, **read_kwargs
    ) -> pd.DataFrame:
        """
        Read dataframes from the given feather file path.

        :param file_path:        The file to read the dataframe from.
        :param unflatten_kwargs: Unflatten keyword arguments for unflattening the read dataframe.
        :param read_kwargs:      Additional keyword arguments to pass to the `read_feather` function.

        :return: The loaded dataframe.
        """
        # Read the feather:
        obj = pd.read_feather(path=file_path, **read_kwargs)

        # Check if it was flattened in packing:
        if unflatten_kwargs is not None:
            obj = cls._unflatten_dataframe(dataframe=obj, **unflatten_kwargs)

        return obj


class _ORCFormatter(_Formatter):
    """
    A static class for managing pandas orc files.
    """

    @classmethod
    def to(
        cls, obj: pd.DataFrame, file_path: str, flatten: bool = True, **to_kwargs
    ) -> dict:
        """
        Save the given dataframe to the orc file path given.

        :param obj:       The dataframe to save.
        :param file_path: The file to save to.
        :param flatten:   Whether to flatten the dataframe before saving. For some formats it is mandatory to enable
                          flattening, otherwise saving and loading the dataframe will cause unexpected behavior
                          especially in case it is multi-level or multi-index. Default to True.
        :param to_kwargs: Additional keyword arguments to pass to the `to_orc` function.

        :return A dictionary of keyword arguments for reading the dataframe from file.
        """
        # Flatten the dataframe (this format have problems saving multi-level dataframes):
        instructions = {}
        if flatten:
            obj, unflatten_kwargs = cls._flatten_dataframe(dataframe=obj)
            instructions["unflatten_kwargs"] = unflatten_kwargs

        # Write to feather:
        obj.to_orc(path=file_path, **to_kwargs)

        return instructions

    @classmethod
    def read(
        cls, file_path: str, unflatten_kwargs: Optional[dict] = None, **read_kwargs
    ) -> pd.DataFrame:
        """
        Read dataframes from the given orc file path.

        :param file_path:        The file to read the dataframe from.
        :param unflatten_kwargs: Unflatten keyword arguments for unflattening the read dataframe.
        :param read_kwargs:      Additional keyword arguments to pass to the `read_orc` function.

        :return: The loaded dataframe.
        """
        # Read the feather:
        obj = pd.read_orc(path=file_path, **read_kwargs)

        # Check if it was flattened in packing:
        if unflatten_kwargs is not None:
            obj = cls._unflatten_dataframe(dataframe=obj, **unflatten_kwargs)

        return obj


class PandasSupportedFormat(SupportedFormat[_Formatter]):
    """
    Library of Pandas formats (file extensions) supported by the Pandas packagers.
    """

    PARQUET = "parquet"
    CSV = "csv"
    H5 = "h5"
    XML = "xml"
    XLSX = "xlsx"
    HTML = "html"
    JSON = "json"
    FEATHER = "feather"
    ORC = "orc"

    _FORMAT_HANDLERS_MAP = {
        PARQUET: _ParquetFormatter,
        CSV: _CSVFormatter,
        H5: _H5Formatter,
        XML: _XMLFormatter,
        XLSX: _XLSXFormatter,
        HTML: _HTMLFormatter,
        JSON: _JSONFormatter,
        FEATHER: _FeatherFormatter,
        ORC: _ORCFormatter,
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

    def pack_result(self, obj: pd.DataFrame, key: str) -> dict:
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

    def pack_file(
        self,
        obj: pd.DataFrame,
        key: str,
        file_format: Optional[str] = None,
        flatten: bool = True,
        **to_kwargs,
    ) -> tuple[Artifact, dict]:
        """
        Pack a dataframe as a file by the given format.

        :param obj:         The series to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is parquet or csv (depends on the column names as
                            parquet cannot be used for non string column names).
        :param flatten:     Whether to flatten the dataframe before saving. For some formats it is mandatory to enable
                            flattening, otherwise saving and loading the dataframe will cause unexpected behavior
                            especially in case it is multi-level or multi-index. Default to True.
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

        # Save to file:
        formatter = PandasSupportedFormat.get_format_handler(fmt=file_format)
        temp_directory = pathlib.Path(tempfile.mkdtemp())
        self.add_future_clearing_path(path=temp_directory)
        file_path = temp_directory / f"{key}.{file_format}"
        read_kwargs = formatter.to(
            obj=obj, file_path=str(file_path), flatten=flatten, **to_kwargs
        )

        # Create the artifact and instructions:
        artifact = Artifact(key=key, src_path=os.path.abspath(file_path))

        return artifact, {"file_format": file_format, "read_kwargs": read_kwargs}

    def pack_dataset(self, obj: pd.DataFrame, key: str, file_format: str = "parquet"):
        """
        Pack a pandas dataframe as a dataset.

        :param obj:         The dataframe to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is parquet.

        :return: The packed artifact and instructions.
        """
        return DatasetArtifact(key=key, df=obj, format=file_format), {}

    def unpack_file(
        self,
        data_item: DataItem,
        file_format: Optional[str] = None,
        read_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Unpack a pandas dataframe from file.

        :param data_item:   The data item to unpack.
        :param file_format: The file format to use for reading the series. Default is None - will be read by the file
                            extension.
        :param read_kwargs: Keyword arguments to pass to the read of the formatter.

        :return: The unpacked series.
        """
        # Get the file:
        file_path = self.get_data_item_local_path(data_item=data_item)

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
        if read_kwargs is None:
            read_kwargs = {}
        return formatter.read(file_path=file_path, **read_kwargs)

    def unpack_dataset(self, data_item: DataItem):
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
                obj[PandasDataFramePackager._prepare_result(obj=key)] = (
                    PandasDataFramePackager._prepare_result(obj=value)
                )
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

    def get_supported_artifact_types(self) -> list[str]:
        """
        Get all the supported artifact types on this packager. It will be the same as `PandasDataFramePackager` but
        without the 'dataset' artifact type support.

        :return: A list of all the supported artifact types.
        """
        supported_artifacts = super().get_supported_artifact_types()
        supported_artifacts.remove("dataset")
        return supported_artifacts

    def pack_result(self, obj: pd.Series, key: str) -> dict:
        """
        Pack a series as a result.

        :param obj: The series to pack and log.
        :param key: The result's key.

        :return: The result dictionary.
        """
        return super().pack_result(obj=pd.DataFrame(obj), key=key)

    def pack_file(
        self,
        obj: pd.Series,
        key: str,
        file_format: Optional[str] = None,
        flatten: bool = True,
        **to_kwargs,
    ) -> tuple[Artifact, dict]:
        """
        Pack a series as a file by the given format.

        :param obj:         The series to pack.
        :param key:         The key to use for the artifact.
        :param file_format: The file format to save as. Default is parquet or csv (depends on the column names as
                            parquet cannot be used for non string column names).
        :param flatten:     Whether to flatten the dataframe before saving. For some formats it is mandatory to enable
                            flattening, otherwise saving and loading the dataframe will cause unexpected behavior
                            especially in case it is multi-level or multi-index. Default to True.
        :param to_kwargs:   Additional keyword arguments to pass to the pandas `to_x` functions.

        :return: The packed artifact and instructions.
        """
        # Get the series column name:
        column_name = obj.name

        # Cast to dataframe and call the parent `pack_file`:
        artifact, instructions = super().pack_file(
            obj=pd.DataFrame(obj),
            key=key,
            file_format=file_format,
            flatten=flatten,
            **to_kwargs,
        )

        # Return the artifact with the updated instructions:
        return artifact, {**instructions, "column_name": column_name}

    def unpack_file(
        self,
        data_item: DataItem,
        file_format: Optional[str] = None,
        read_kwargs: Optional[dict] = None,
        column_name: Optional[Union[str, int]] = None,
    ) -> pd.Series:
        """
        Unpack a pandas series from file.

        :param data_item:     The data item to unpack.
        :param file_format:   The file format to use for reading the series. Default is None - will be read by the file
                              extension.
        :param read_kwargs:   Keyword arguments to pass to the read of the formatter.
        :param column_name:   The name of the series column.

        :return: The unpacked series.
        """
        # Read the object:
        obj = super().unpack_file(
            data_item=data_item,
            file_format=file_format,
            read_kwargs=read_kwargs,
        )

        # Cast the dataframe into a series:
        if len(obj.columns) != 1:
            raise MLRunInvalidArgumentError(
                f"The data item received is of a `pandas.DataFrame` with more than one column: "
                f"{', '.join(obj.columns)}. Hence it cannot be turned into a `pandas.Series`."
            )
        obj = obj[obj.columns[0]]

        # Edit the column name (if `read_kwargs` is not None we can be sure it is a packed file artifact, so the column
        # name, even if None, should be set to restore the object as it was):
        if read_kwargs is not None:
            obj.name = column_name

        return obj
