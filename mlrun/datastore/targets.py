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
import os
import sys
from copy import copy
from typing import Dict
import mlrun
from mlrun.utils import now_date

from mlrun.model import DataTargetBase, DataTarget
from .v3io import parse_v3io_path
from .utils import store_path_to_spark


class TargetTypes:
    csv = "csv"
    parquet = "parquet"
    nosql = "nosql"
    tsdb = "tsdb"
    stream = "stream"
    dataframe = "dataframe"
    custom = "custom"

    @staticmethod
    def all():
        return [
            TargetTypes.csv,
            TargetTypes.parquet,
            TargetTypes.nosql,
            TargetTypes.tsdb,
            TargetTypes.stream,
            TargetTypes.dataframe,
            TargetTypes.custom,
        ]


def default_target_names():
    targets = mlrun.mlconf.feature_store.default_targets
    return [target.strip() for target in targets.split(",")]


def get_default_targets():
    """initialize the default feature set targets list"""
    return [
        DataTargetBase(target, name=str(target)) for target in default_target_names()
    ]


def add_target_states(graph, resource, targets, to_df=False, final_state=None):
    """add the target states to the graph"""
    targets = targets or []
    key_column = resource.spec.entities[0].name
    timestamp_key = resource.spec.timestamp_key
    features = resource.spec.features
    table = None

    for target in targets:
        driver = get_target_driver(target, resource)
        table = driver.get_table_object() or table
        driver.update_resource_status()
        driver.add_writer_state(
            graph,
            target.after_state or final_state,
            features=features,
            key_column=key_column,
            timestamp_key=timestamp_key,
        )
    if to_df:
        # add dataframe target, will return a dataframe
        driver = DFTarget()
        driver.add_writer_state(
            graph,
            final_state,
            features=features,
            key_column=key_column,
            timestamp_key=timestamp_key,
        )

    return table


offline_lookup_order = [TargetTypes.parquet, TargetTypes.csv]
online_lookup_order = [TargetTypes.nosql]


def get_offline_target(featureset, start_time=None, name=None):
    """return an optimal offline feature set target"""
    # todo: take status, start_time and lookup order into account
    for target in featureset.status.targets:
        driver = kind_to_driver[target.kind]
        if driver.is_offline and (not name or name == target.name):
            return get_target_driver(target, featureset)
    return None


def get_online_target(resource):
    """return an optimal online feature set target"""
    # todo: take lookup order into account
    for target in resource.status.targets:
        driver = kind_to_driver[target.kind]
        if driver.is_online:
            return get_target_driver(target, resource)
    return None


def get_target_driver(target_spec, resource=None):
    if isinstance(target_spec, dict):
        target_spec = DataTargetBase.from_dict(target_spec)
    driver_class = kind_to_driver[target_spec.kind]
    return driver_class.from_spec(target_spec, resource)


class BaseStoreTarget(DataTargetBase):
    """base target storage driver, used to materialize feature set/vector data"""

    kind = ""
    is_table = False
    suffix = ""
    is_online = False
    is_offline = False
    support_spark = False
    support_storey = False

    def __init__(
        self,
        name: str = "",
        path=None,
        attributes: Dict[str, str] = None,
        after_state=None,
    ):
        self.name = name
        self.path = str(path) if path is not None else None
        self.after_state = after_state
        self.attributes = attributes or {}

        self._target = None
        self._resource = None
        self._secrets = {}

    def _get_store(self):
        store, _ = mlrun.store_manager.get_or_create_store(self._target_path)
        return store

    def write_dataframe(self, df, key_column=None, timestamp_key=None, **kwargs):
        if hasattr(df, "rdd"):
            options = self.get_spark_options(key_column, timestamp_key)
            options.update(kwargs)
            df.write.mode("overwrite").save(**options)
        else:
            target_path = self._target_path
            fs = self._get_store().get_filesystem(False)
            if fs.protocol == "file":
                dir = os.path.dirname(target_path)
                if dir:
                    os.makedirs(dir, exist_ok=True)
            self._write_dataframe(df, fs, target_path, **kwargs)
            try:
                return fs.size(target_path)
            except Exception:
                return None

    @staticmethod
    def _write_dataframe(df, fs, target_path, **kwargs):
        raise NotImplementedError()

    def set_secrets(self, secrets):
        self._secrets = secrets

    def set_resource(self, resource):
        self._resource = resource

    @classmethod
    def from_spec(cls, spec: DataTargetBase, resource=None):
        """create target driver from target spec or other target driver"""
        driver = cls()
        driver.name = spec.name
        driver.path = spec.path
        driver.attributes = spec.attributes
        driver._resource = resource
        return driver

    def get_table_object(self):
        """get storey Table object"""
        return None

    @property
    def _target_path(self):
        """return the actual/computed target path"""
        return self.path or _get_target_path(self, self._resource)

    def update_resource_status(self, status="", producer=None, is_dir=None, size=None):
        """update the data target status"""
        self._target = self._target or DataTarget(
            self.kind, self.name, self._target_path
        )
        target = self._target
        target.is_dir = is_dir
        target.status = status or target.status or "created"
        target.updated = now_date().isoformat()
        target.size = size
        target.producer = producer or target.producer
        self._resource.status.update_target(target)
        return target

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        """add storey writer state to graph"""
        raise NotImplementedError()

    def as_df(self, columns=None, df_module=None):
        """return the target data as dataframe"""
        return mlrun.get_dataitem(self._target_path).as_df(
            columns=columns, df_module=df_module
        )

    def get_spark_options(self, key_column=None, timestamp_key=None):
        # options used in spark.read.load(**options)
        raise NotImplementedError()


class ParquetTarget(BaseStoreTarget):
    kind = TargetTypes.parquet
    suffix = ".parquet"
    is_offline = True
    support_spark = True
    support_storey = True

    @staticmethod
    def _write_dataframe(df, fs, target_path, **kwargs):
        with fs.open(target_path, "wb") as fp:
            df.to_parquet(fp, **kwargs)

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        column_list = list(features.keys())
        if timestamp_key:
            column_list = [timestamp_key] + column_list

        graph.add_step(
            name="WriteToParquet",
            after=after,
            graph_shape="cylinder",
            class_name="storey.WriteToParquet",
            path=self._target_path,
            columns=column_list,
            index_cols=key_column,
            storage_options=self._get_store().get_storage_options(),
        )

    def get_spark_options(self, key_column=None, timestamp_key=None):
        return {
            "path": store_path_to_spark(self._target_path),
            "format": "parquet",
        }


class CSVTarget(BaseStoreTarget):
    kind = TargetTypes.csv
    suffix = ".csv"
    is_offline = True
    support_spark = True
    support_storey = True

    @staticmethod
    def _write_dataframe(df, fs, target_path, **kwargs):
        mode = "wb"
        # We generally prefer to open in a binary mode so that different encodings could be used, but pandas had a bug
        # with such files until version 1.2.0, in this version they dropped support for python 3.6.
        # So only for python 3.6 we're using text mode which might prevent some features
        if sys.version_info[0] == 3 and sys.version_info[1] == 6:
            mode = "wt"
        with fs.open(target_path, mode) as fp:
            df.to_csv(fp, **kwargs)

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        column_list = list(features.keys())
        if timestamp_key:
            column_list = [timestamp_key] + column_list

        graph.add_step(
            name="WriteToCSV",
            after=after,
            graph_shape="cylinder",
            class_name="storey.WriteToCSV",
            path=self._target_path,
            columns=column_list,
            header=True,
            index_cols=key_column,
            storage_options=self._get_store().get_storage_options(),
        )

    def get_spark_options(self, key_column=None, timestamp_key=None):
        return {
            "path": store_path_to_spark(self._target_path),
            "format": "csv",
            "header": "true",
        }


class NoSqlTarget(BaseStoreTarget):
    kind = TargetTypes.nosql
    is_table = True
    is_online = True
    support_spark = True
    support_storey = True

    def get_table_object(self):
        from storey import Table, V3ioDriver

        # TODO use options/cred
        endpoint, uri = parse_v3io_path(self._target_path)
        return Table(uri, V3ioDriver(webapi=endpoint))

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        table = self._resource.uri
        column_list = [
            key for key, feature in features.items() if not feature.aggregate
        ]
        graph.add_step(
            name="WriteToTable",
            after=after,
            graph_shape="cylinder",
            class_name="storey.WriteToTable",
            columns=column_list,
            table=table,
        )

    def get_spark_options(self, key_column=None, timestamp_key=None):
        return {
            "path": store_path_to_spark(self._target_path),
            "format": "io.iguaz.v3io.spark.sql.kv",
            "key": key_column,
        }

    def as_df(self, columns=None, df_module=None):
        raise NotImplementedError()


class StreamTarget(BaseStoreTarget):
    kind = TargetTypes.stream
    is_table = False
    is_online = False
    support_spark = False
    support_storey = True

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        from storey import V3ioDriver

        endpoint, uri = parse_v3io_path(self._target_path)
        column_list = list(features.keys())
        if timestamp_key:
            column_list = [timestamp_key] + column_list
        graph.add_step(
            name="WriteToStream",
            after=after,
            graph_shape="cylinder",
            class_name="storey.WriteToV3IOStream",
            columns=column_list,
            storage=V3ioDriver(webapi=endpoint),
            stream_path=uri,
            **self.attributes,
        )

    def as_df(self, columns=None, df_module=None):
        raise NotImplementedError()


class CustomTarget(BaseStoreTarget):
    kind = "custom"
    is_table = False
    is_online = False
    support_spark = False
    support_storey = True

    def __init__(
        self, class_name: str, name: str = "", after_state=None, **attributes,
    ):
        attributes = attributes or {}
        attributes["class_name"] = class_name
        super().__init__(name, "", attributes, after_state=after_state)

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        attributes = copy(self.attributes)
        class_name = attributes.pop("class_name")
        graph.add_step(
            name=self.name,
            after=after,
            graph_shape="cylinder",
            class_name=class_name,
            **attributes,
        )


class DFTarget(BaseStoreTarget):
    support_storey = True

    def __init__(self):
        self.name = "dataframe"
        self._df = None

    def set_df(self, df):
        self._df = df

    def update_resource_status(self, status="", producer=None, is_dir=None):
        pass

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        # todo: column filter
        graph.add_step(
            name="WriteToDataFrame",
            after=after,
            graph_shape="cylinder",
            class_name="storey.ReduceToDataFrame",
            index=key_column,
            insert_key_column_as=key_column,
            insert_time_column_as=timestamp_key,
        )

    def as_df(self, columns=None, df_module=None):
        return self._df


kind_to_driver = {
    TargetTypes.parquet: ParquetTarget,
    TargetTypes.csv: CSVTarget,
    TargetTypes.nosql: NoSqlTarget,
    TargetTypes.dataframe: DFTarget,
    TargetTypes.stream: StreamTarget,
    TargetTypes.custom: CustomTarget,
}


def _get_target_path(driver, resource):
    """return the default target path given the resource and target kind"""
    kind = driver.kind
    suffix = driver.suffix
    kind_prefix = (
        "sets"
        if resource.kind == mlrun.api.schemas.ObjectKind.feature_set
        else "vectors"
    )
    name = resource.metadata.name
    version = resource.metadata.tag
    project = resource.metadata.project or mlrun.mlconf.default_project
    data_prefixes = mlrun.mlconf.feature_store.data_prefixes
    data_prefix = getattr(data_prefixes, kind, None)
    if not data_prefix:
        data_prefix = data_prefixes.default
    data_prefix = data_prefix.format(project=project, kind=kind)
    # todo: handle ver tag changes, may need to copy files?
    name = f"{name}-{version or 'latest'}"
    return f"{data_prefix}/{kind_prefix}/{name}{suffix}"
