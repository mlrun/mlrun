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
from typing import Dict
import mlrun

from .model.base import DataSource
from ..datastore import store_path_to_spark


def get_source_from_dict(source):
    kind = source["kind"] or ""
    return source_kind_to_driver[kind].from_dict(source)


def get_source_step(source, key_column=None, time_column=None):
    """initialize the source driver"""
    if hasattr(source, "to_csv"):
        source = DataFrameSource(source)
    if not key_column and not source.key_column:
        raise mlrun.errors.MLRunInvalidArgumentError("key column is not defined")
    return source.to_step(key_column, time_column)


class BaseSourceDriver(DataSource):
    support_spark = False
    support_storey = False

    def to_step(self, key_column=None, time_column=None):
        import storey

        return storey.Source()

    def get_table_object(self):
        """get storey Table object"""
        return None

    def to_dataframe(self):
        return mlrun.store_manager.object(url=self.path).as_df()

    def to_spark_df(self, session, named_view=False):
        if self.support_spark:
            df = session.read.load(**self.get_spark_options())
            if named_view:
                df.createOrReplaceTempView(self.name)
            return df
        raise NotImplementedError()

    def get_spark_options(self):
        # options used in spark.read.load(**options)
        raise NotImplementedError()


class CSVSource(BaseSourceDriver):
    kind = "csv"
    support_storey = True
    support_spark = True

    def __init__(
        self,
        name: str = "",
        path: str = None,
        attributes: Dict[str, str] = None,
        key_column: str = None,
        time_column: str = None,
        schedule: str = None,
    ):
        super().__init__(name, path, attributes, key_column, time_column, schedule)

    def to_step(self, key_column=None, time_column=None):
        import storey

        attributes = self.attributes or {}
        return storey.ReadCSV(
            paths=self.path,
            header=True,
            build_dict=True,
            key_field=self.key_column or key_column,
            timestamp_field=self.time_column or time_column,
            **attributes,
        )

    def get_spark_options(self):
        return {
            "path": store_path_to_spark(self.path),
            "format": "csv",
            "header": "true",
            "inferSchema": "true",
        }


class ParquetSource(BaseSourceDriver):
    kind = "csv"
    support_storey = True
    support_spark = True

    def __init__(
        self,
        name: str = "",
        path: str = None,
        attributes: Dict[str, str] = None,
        key_column: str = None,
        time_column: str = None,
        schedule: str = None,
    ):
        super().__init__(name, path, attributes, key_column, time_column, schedule)

    def to_step(self, key_column=None, time_column=None):
        import storey

        attributes = self.attributes or {}
        return storey.ReadParquet(
            paths=self.path,
            key_column=self.key_column or key_column,
            time_column=self.time_column or time_column,
            **attributes,
        )

    def get_spark_options(self):
        return {
            "path": store_path_to_spark(self.path),
            "format": "parquet",
        }


class DataFrameSource:
    support_storey = True

    def __init__(self, df, key_column=None, time_column=None):
        self._df = df
        self.key_column = key_column
        self.time_column = time_column

    def to_step(self, key_column=None, time_column=None):
        import storey

        return storey.DataframeSource(
            dfs=self._df,
            key_column=self.key_column or key_column,
            time_column=self.time_column or time_column,
        )

    def to_dataframe(self):
        return self._df


# map of sources (exclude DF source which is not serializable)
source_kind_to_driver = {
    "": BaseSourceDriver,
    "csv": CSVSource,
}
