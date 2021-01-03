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
from mlrun.utils import now_date

from mlrun.run import get_dataitem
from mlrun.config import config as mlconf

from .model.base import (
    DataTargetSpec,
    TargetTypes,
    DataTarget,
    store_config,
    ResourceKinds,
)
from mlrun.datastore.v3io import v3io_path


def get_default_targets():
    """initialize the default feature set targets list"""
    return [
        DataTargetSpec(target, name=str(target))
        for target in store_config.default_targets
    ]


def add_target_states(graph, resource, targets, to_df=False, final_state=None):
    """add the target states to the graph"""
    targets = targets or []
    key_column = resource.spec.entities[0].name
    timestamp_key = resource.spec.timestamp_key
    features = resource.spec.features
    table = None

    for target in targets:
        driver = kind_to_driver[target.kind](resource, target)
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
        driver = DFStore(resource)
        driver.add_writer_state(
            graph,
            final_state,
            features=features,
            key_column=key_column,
            timestamp_key=timestamp_key,
        )

    return table


def update_target_status(featureset, status, producer):
    for target in featureset.spec.targets:
        target.status = status
        target.producer = producer


offline_lookup_order = [TargetTypes.parquet]
online_lookup_order = [TargetTypes.nosql]


def get_offline_target(featureset, start_time=None, name=None):
    # todo: take status, start_time and lookup order into account
    for target in featureset.status.targets:
        driver = kind_to_driver[target.kind]
        if driver.is_offline and (not name or name == target.name):
            return target, driver(featureset, target)
    return None, None


def get_online_target(featureset):
    for target in featureset.status.targets:
        driver = kind_to_driver[target.kind]
        if driver.is_online:
            return target, driver(featureset, target)
    return None, None


class BaseStoreDriver:
    kind = None
    is_table = False
    suffix = ""
    is_online = False
    is_offline = False

    def __init__(self, resource, target_spec: DataTargetSpec):
        self.name = target_spec.name
        self.target_path = target_spec.path or _get_target_path(self, resource)
        self.attributes = target_spec.attributes
        self.target = None
        self.resource = resource

    def get_table_object(self):
        """get storey Table object"""
        return None

    def update_resource_status(self, status="", producer=None):
        """update the data target status"""
        self.target = self.target or DataTarget(self.kind, self.name, self.target_path)
        self.target.status = status or self.target.status
        self.target.updated = now_date().isoformat()
        self.target.producer = producer or self.target.producer
        self.resource.status.update_target(self.target)

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        pass

    def as_df(self, columns=None, df_module=None):
        return get_dataitem(self.target_path).as_df(
            columns=columns, df_module=df_module
        )


class ParquetStore(BaseStoreDriver):
    kind = TargetTypes.parquet
    suffix = ".parquet"
    is_offline = True

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
            path=self.target_path,
            columns=column_list,
            index_cols=key_column,
        )


class CSVStore(BaseStoreDriver):
    kind = TargetTypes.csv
    suffix = ".csv"
    is_offline = True

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
            path=self.target_path,
            columns=column_list,
            header=True,
            index_cols=key_column,
        )


class NoSqlStore(BaseStoreDriver):
    kind = TargetTypes.nosql
    is_table = True
    is_online = True

    def __init__(self, resource, target_spec: DataTargetSpec):
        self.name = target_spec.name
        self.target_path = target_spec.path or _get_target_path(self, resource)
        self.resource = resource
        self.attributes = target_spec.attributes
        self.target = None

    def get_table_object(self):
        from storey import Table, V3ioDriver

        # TODO use options/cred
        endpoint, uri = v3io_path(self.target_path)
        return Table(uri, V3ioDriver(webapi=endpoint))

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        table = self.resource.uri()
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

    def as_df(self, columns=None, df_module=None):
        raise NotImplementedError()


class DFStore(BaseStoreDriver):
    def __init__(self, resource, target_spec: DataTargetSpec = None):
        self.name = "dataframe"
        self.target = None
        self._df = None

    def set_df(self, df):
        self._df = df

    def update_resource_status(self, status="", producer=None):
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
    TargetTypes.parquet: ParquetStore,
    TargetTypes.csv: CSVStore,
    TargetTypes.nosql: NoSqlStore,
    TargetTypes.dataframe: DFStore,
}


def _get_target_path(driver, resource):
    kind = driver.kind
    suffix = driver.suffix
    kind_prefix = "sets" if resource.kind == ResourceKinds.FeatureSet else "vectors"
    name = resource.metadata.name
    version = resource.metadata.tag
    project = resource.metadata.project or mlconf.default_project
    data_prefix = store_config.data_prefixes.get(kind, None)
    if not data_prefix:
        data_prefix = store_config.data_prefixes["default"]
    data_prefix = data_prefix.format(project=project, kind=kind)
    if version:
        name = f"{name}-{version}"
    return f"{data_prefix}/{kind_prefix}/{name}{suffix}"
