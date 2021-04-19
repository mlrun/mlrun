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
import typing
from copy import copy

import mlrun
import mlrun.utils.helpers
from mlrun.config import config
from mlrun.model import DataTarget, DataTargetBase
from mlrun.utils import now_date
from mlrun.utils.v3io_clients import get_frames_client

from ..platforms.iguazio import split_path
from .utils import store_path_to_spark
from .v3io import parse_v3io_path


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


def validate_target_placement(graph, final_step, targets):
    if final_step or graph.is_empty():
        return True
    for target in targets:
        if not target.after_state:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "writer step location is undetermined due to graph branching"
                ", set the target .after_state attribute or the graph .final_state"
            )


def add_target_states(graph, resource, targets, to_df=False, final_state=None):
    """add the target states to the graph"""
    targets = targets or []
    key_columns = list(resource.spec.entities.keys())
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
            features=features if not target.after_state else None,
            key_columns=key_columns,
            timestamp_key=timestamp_key,
        )
    if to_df:
        # add dataframe target, will return a dataframe
        driver = DFTarget()
        driver.add_writer_state(
            graph,
            final_state,
            features=features,
            key_columns=key_columns,
            timestamp_key=timestamp_key,
        )

    return table


def _get_column_list(features, timestamp_key, key_columns):
    column_list = None
    if features:
        column_list = list(features.keys())
        if timestamp_key and timestamp_key not in column_list:
            column_list = [timestamp_key] + column_list
        if key_columns:
            for key in reversed(key_columns):
                if key not in column_list:
                    column_list.insert(0, key)
    return column_list


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
        attributes: typing.Dict[str, str] = None,
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

    def write_dataframe(
        self, df, key_column=None, timestamp_key=None, **kwargs,
    ) -> typing.Optional[int]:
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
        self, graph, after, features, key_columns=None, timestamp_key=None
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
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        column_list = _get_column_list(
            features=features, timestamp_key=timestamp_key, key_columns=None
        )

        graph.add_step(
            name="WriteToParquet",
            after=after,
            graph_shape="cylinder",
            class_name="storey.WriteToParquet",
            path=self._target_path,
            columns=column_list,
            index_cols=key_columns,
            storage_options=self._get_store().get_storage_options(),
            **self.attributes,
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
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        column_list = _get_column_list(
            features=features, timestamp_key=timestamp_key, key_columns=key_columns
        )

        graph.add_step(
            name="WriteToCSV",
            after=after,
            graph_shape="cylinder",
            class_name="storey.WriteToCSV",
            path=self._target_path,
            columns=column_list,
            header=True,
            index_cols=key_columns,
            storage_options=self._get_store().get_storage_options(),
            **self.attributes,
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
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        table = self._resource.uri
        column_list = None
        if features:
            column_list = [
                key for key, feature in features.items() if not feature.aggregate
            ]
            for key in reversed(key_columns):
                if key not in column_list:
                    column_list.insert(0, key)
        graph.add_step(
            name="WriteToTable",
            after=after,
            graph_shape="cylinder",
            class_name="storey.WriteToTable",
            columns=column_list,
            table=table,
            **self.attributes,
        )

    def get_spark_options(self, key_column=None, timestamp_key=None):
        return {
            "path": store_path_to_spark(self._target_path),
            "format": "io.iguaz.v3io.spark.sql.kv",
            "key": key_column,
        }

    def as_df(self, columns=None, df_module=None):
        raise NotImplementedError()

    def write_dataframe(self, df, key_column=None, timestamp_key=None, **kwargs):
        if hasattr(df, "rdd"):
            options = self.get_spark_options(key_column, timestamp_key)
            options.update(kwargs)
            df.write.mode("overwrite").save(**options)
        else:
            access_key = self._secrets.get(
                "V3IO_ACCESS_KEY", os.getenv("V3IO_ACCESS_KEY")
            )

            _, path_with_container = parse_v3io_path(self._target_path)
            container, path = split_path(path_with_container)

            frames_client = get_frames_client(
                token=access_key, address=config.v3io_framesd, container=container
            )

            frames_client.write("kv", path, df, index_cols=key_column, **kwargs)


class StreamTarget(BaseStoreTarget):
    kind = TargetTypes.stream
    is_table = False
    is_online = False
    support_spark = False
    support_storey = True

    def add_writer_state(
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        from storey import V3ioDriver

        endpoint, uri = parse_v3io_path(self._target_path)
        column_list = _get_column_list(
            features=features, timestamp_key=timestamp_key, key_columns=key_columns
        )

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


class TSDBTarget(BaseStoreTarget):
    kind = TargetTypes.tsdb
    is_table = False
    is_online = False
    support_spark = False
    support_storey = True

    def add_writer_state(
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        endpoint, uri = parse_v3io_path(self._target_path)
        if not timestamp_key:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "feature set timestamp_key must be specified for TSDBTarget writer"
            )

        column_list = _get_column_list(
            features=features, timestamp_key=None, key_columns=key_columns
        )

        graph.add_step(
            name="WriteToTSDB",
            class_name="storey.WriteToTSDB",
            after=after,
            graph_shape="cylinder",
            path=uri,
            time_col=timestamp_key,
            index_cols=key_columns,
            columns=column_list,
            **self.attributes,
        )

    def as_df(self, columns=None, df_module=None):
        raise NotImplementedError()

    def write_dataframe(self, df, key_column=None, timestamp_key=None, **kwargs):
        access_key = self._secrets.get("V3IO_ACCESS_KEY", os.getenv("V3IO_ACCESS_KEY"))

        new_index = []
        if timestamp_key:
            new_index.append(timestamp_key)
        if key_column:
            if isinstance(key_column, str):
                key_column = [key_column]
            new_index.extend(key_column)

        _, path_with_container = parse_v3io_path(self._target_path)
        container, path = split_path(path_with_container)

        frames_client = get_frames_client(
            token=access_key, address=config.v3io_framesd, container=container,
        )

        frames_client.write(
            "tsdb", path, df, index_cols=new_index if new_index else None, **kwargs
        )


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
        self, graph, after, features, key_columns=None, timestamp_key=None
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
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        # todo: column filter
        graph.add_step(
            name="WriteToDataFrame",
            after=after,
            graph_shape="cylinder",
            class_name="storey.ReduceToDataFrame",
            index=key_columns,
            insert_key_column_as=key_columns,
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
    TargetTypes.tsdb: TSDBTarget,
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
