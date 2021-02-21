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
from copy import copy
from typing import Dict
import mlrun

from ..model import DataSource
from .utils import store_path_to_spark
from ..utils import get_class


def get_source_from_dict(source):
    kind = source.get("kind", "")
    if not kind:
        return None
    return source_kind_to_driver[kind].from_dict(source)


def get_source_step(source, key_field=None, time_field=None):
    """initialize the source driver"""
    if hasattr(source, "to_csv"):
        source = DataFrameSource(source)
    if not key_field and not source.key_field:
        raise mlrun.errors.MLRunInvalidArgumentError("key column is not defined")
    return source.to_step(key_field, time_field)


class BaseSourceDriver(DataSource):
    support_spark = False
    support_storey = False

    def _get_store(self):
        store, _ = mlrun.store_manager.get_or_create_store(self.path)
        return store

    def to_step(self, key_field=None, time_field=None):
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
        key_field: str = None,
        time_field: str = None,
        schedule: str = None,
    ):
        super().__init__(name, path, attributes, key_field, time_field, schedule)

    def to_step(self, key_field=None, time_field=None):
        import storey

        attributes = self.attributes or {}
        return storey.ReadCSV(
            paths=self.path,
            header=True,
            build_dict=True,
            key_field=self.key_field or key_field,
            timestamp_field=self.time_field or time_field,
            storage_options=self._get_store().get_storage_options(),
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
    kind = "parquet"
    support_storey = True
    support_spark = True

    def __init__(
        self,
        name: str = "",
        path: str = None,
        attributes: Dict[str, str] = None,
        key_field: str = None,
        time_field: str = None,
        schedule: str = None,
    ):
        super().__init__(name, path, attributes, key_field, time_field, schedule)

    def to_step(self, key_field=None, time_field=None):
        import storey

        attributes = self.attributes or {}
        return storey.ReadParquet(
            paths=self.path,
            key_field=self.key_field or key_field,
            time_field=self.time_field or time_field,
            storage_options=self._get_store().get_storage_options(),
            **attributes,
        )

    def get_spark_options(self):
        return {
            "path": store_path_to_spark(self.path),
            "format": "parquet",
        }


class CustomSource(BaseSourceDriver):
    kind = "custom"
    support_storey = True
    support_spark = False

    def __init__(
        self,
        class_name: str = None,
        name: str = "",
        schedule: str = None,
        **attributes,
    ):
        attributes = attributes or {}
        attributes["class_name"] = class_name
        super().__init__(name, "", attributes, schedule=schedule)

    def to_step(self, key_field=None, time_field=None):
        attributes = copy(self.attributes)
        class_name = attributes.pop("class_name")
        class_object = get_class(class_name)
        return class_object(**attributes,)


class DataFrameSource:
    support_storey = True

    def __init__(self, df, key_field=None, time_field=None):
        self._df = df
        self.key_field = key_field
        self.time_field = time_field

    def to_step(self, key_field=None, time_field=None):
        import storey

        return storey.DataframeSource(
            dfs=self._df,
            key_field=self.key_field or key_field,
            time_field=self.time_field or time_field,
        )

    def to_dataframe(self):
        return self._df


class OnlineSource(BaseSourceDriver):
    """online data source spec"""

    _dict_fields = [
        "kind",
        "name",
        "path",
        "attributes",
        "online",
        "workers",
    ]
    kind = ""

    def __init__(
        self,
        name: str = None,
        path: str = None,
        attributes: Dict[str, str] = None,
        workers: int = None,
    ):
        super().__init__(name, path, attributes)
        self.online = True
        self.workers = workers


class HttpSource(OnlineSource):
    kind = "http"


# map of sources (exclude DF source which is not serializable)
source_kind_to_driver = {
    "": BaseSourceDriver,
    "csv": CSVSource,
    "parquet": ParquetSource,
    "http": HttpSource,
    "custom": CustomSource,
}
