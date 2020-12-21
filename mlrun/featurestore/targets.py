from copy import copy
from urllib.parse import urlparse

from mlrun.run import get_dataitem
from storey import Table, V3ioDriver
from mlrun.config import config as mlconf

from .model import DataTargetSpec, TargetTypes, DataTarget, store_config, ResourceKinds


def init_store_driver(resource, target):
    driver = kind_to_driver[target.kind](resource, target)
    table = driver.get_table_object()
    driver.update_resource_status(resource)
    target.driver = driver
    return table


def init_featureset_targets(featureset):
    targets = featureset.spec.targets
    table = None

    if not targets:
        for target in store_config.default_targets:
            target_obj = targets.update(DataTargetSpec(target), str(target))
            table = init_store_driver(featureset, target_obj) or table
    else:
        for target in targets:
            table = init_store_driver(featureset, target) or table
    return table


def add_target_states(graph, resource, targets, to_df=False):
    if len(graph.states) > 0:
        after = resource.spec.get_final_state()
    else:
        graph.add_step(name="_in", handler="(event)", after="$start")
        after = "_in"
    targets = targets or []
    key_column = resource.spec.entities[0].name
    timestamp_key = resource.spec.timestamp_key
    features = resource.spec.features

    for target in targets:
        target.driver.add_writer_state(
            graph,
            target.after_state or after,
            features=features,
            key_column=key_column,
            timestamp_key=timestamp_key,
        )
    if to_df:
        driver = DFStore(resource)
        driver.add_writer_state(
            graph,
            after,
            features=features,
            key_column=key_column,
            timestamp_key=timestamp_key,
        )


def update_target_status(featureset, status, producer):
    pass


offline_lookup_order = [TargetTypes.parquet]
online_lookup_order = [TargetTypes.nosql]


def get_offline_target(featureset, start_time=None, name=None):
    # todo take status, start_time and lookup order into account
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

    def get_table_object(self):
        return None

    def update_resource_status(self, resource, status="", producer=None):
        self.target = self.target or DataTarget(self.kind, self.name, self.target_path)
        self.target.status = status or self.target.status
        self.target.producer = producer or self.target.producer
        resource.status.update_target(self.target)

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        pass

    def source_to_step(self, source):
        return None

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
            shape="cylinder",
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
            shape="cylinder",
            class_name="storey.WriteToCSV",
            path=self.target_path,
            columns=column_list,
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
        # TODO use options/cred
        return Table(nosql_path(self.target_path), V3ioDriver())

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
            shape="cylinder",
            class_name="storey.WriteToTable",
            columns=column_list,
            table=table,
        )


class DFStore(BaseStoreDriver):
    def __init__(self, resource, target_spec: DataTargetSpec = None):
        self.name = "dataframe"
        self.target = None

    def update_resource_status(self, resource, status="", producer=None):
        pass

    def add_writer_state(
        self, graph, after, features, key_column=None, timestamp_key=None
    ):
        # todo: column filter
        graph.add_step(
            name="WriteToDataFrame",
            after=after,
            shape="cylinder",
            class_name="storey.ReduceToDataFrame",
            index=key_column,
            insert_key_column_as=key_column,
            insert_time_column_as=timestamp_key,
        )


kind_to_driver = {
    TargetTypes.parquet: ParquetStore,
    TargetTypes.csv: CSVStore,
    TargetTypes.nosql: NoSqlStore,
    TargetTypes.dataframe: DFStore,
}


def nosql_path(url):
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme.lower()
    if scheme != "v3io":
        raise ValueError(
            "url must start with v3io://[host]/{container}/{path}, got " + url
        )

    endpoint = parsed_url.hostname
    if parsed_url.port:
        endpoint += ":{}".format(parsed_url.port)
    # todo: use endpoint
    return parsed_url.path.strip("/") + "/"


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
