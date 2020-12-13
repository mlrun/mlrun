from copy import copy
from typing import List
from urllib.parse import urlparse
from storey import Table, V3ioDriver
from mlrun.config import config as mlconf

from .model import DataTargetSpec, TargetTypes, DataTarget, store_config


def init_target(featureset, target, tables=None):
    driver = kind_to_driver[target.kind](featureset, target)
    driver.init_table(tables)
    driver.update_featureset_status()
    target.driver = driver


def init_featureset_targets(featureset, tables):
    targets = featureset.spec.targets

    if not targets:
        defaults = copy(store_config.default_targets)
        for target in defaults:
            target_obj = targets.update(DataTargetSpec(target), str(target))
            init_target(featureset, target_obj, tables)
    else:
        for target in targets:
            init_target(featureset, target, tables)


def add_target_states(graph, featureset, targets, to_df=False):
    after = featureset.spec.get_final_state()
    targets = targets or []
    for target in targets:
        target.driver.add_writer_state(graph, target.after_state or after)
    if to_df:
        driver = DFTarget(featureset)
        driver.add_writer_state(graph, after)


def update_target_status(featureset, status, producer):
    pass


offline_lookup_order = [TargetTypes.parquet]
online_lookup_order = [TargetTypes.nosql]


def get_offline_target(featureset, start_time=None):
    # todo take status, start_time and lookup order into account
    for target in featureset.status.targets:
        driver = kind_to_driver[target.kind]
        if driver.is_offline:
            return target, driver


def get_online_target(featureset):
    for target in featureset.status.targets:
        driver = kind_to_driver[target.kind]
        if driver.is_online:
            return target, driver


class BaseTargetDriver:
    kind = None
    is_table = False
    suffix = ""
    is_online = False
    is_offline = False

    def __init__(self, featureset, target_spec: DataTargetSpec):
        self.name = target_spec.name
        self.target_path = target_spec.path or _get_target_path(
            self.kind, featureset, self.suffix
        )
        self.featureset = featureset
        self.target = None

    @staticmethod
    def get_table_object(target_path):
        pass

    def init_table(self, tables, default=True):
        if self.is_table and tables is not None:
            table = self.get_table_object(self.target_path)
            tables[self.featureset.uri()] = table
            if default:
                tables["."] = table

    def update_featureset_status(self, status="", producer=None):
        self.target = self.target or DataTarget(self.kind, self.name, self.target_path)
        self.target.status = status or self.target.status
        self.target.producer = producer or self.target.producer
        self.featureset.status.update_target(self.target)

    def add_writer_state(self, graph, after):
        pass


class ParquetTarget(BaseTargetDriver):
    kind = TargetTypes.parquet
    suffix = ".parquet"
    is_offline = True

    def add_writer_state(self, graph, after):
        key_column = self.featureset.spec.entities[0].name
        timestamp_key = self.featureset.spec.timestamp_key
        column_list = list(self.featureset.spec.features.keys())
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


class CSVTarget(BaseTargetDriver):
    kind = TargetTypes.csv
    suffix = ".csv"
    is_offline = True

    def add_writer_state(self, graph, after):
        key_column = self.featureset.spec.entities[0].name
        timestamp_key = self.featureset.spec.timestamp_key
        column_list = list(self.featureset.spec.features.keys())
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


class NoSqlTarget(BaseTargetDriver):
    kind = TargetTypes.nosql
    is_table = True
    is_online = True

    def __init__(self, featureset, target_spec: DataTargetSpec):
        self.name = target_spec.name
        self.target_path = nosql_path(
            target_spec.path or _get_target_path(self.kind, featureset, self.suffix)
        )
        self.featureset = featureset
        self.target = None

    @staticmethod
    def get_table_object(target_path, options=None):
        # TODO use options/cred
        return Table(target_path, V3ioDriver())

    def add_writer_state(self, graph, after):
        table = self.featureset.uri()
        column_list = [
            key
            for key, feature in self.featureset.spec.features.items()
            if not feature.aggregate
        ]
        graph.add_step(
            name="WriteToTable",
            after=after,
            shape="cylinder",
            class_name="storey.WriteToTable",
            columns=column_list,
            table=table,
        )


class DFTarget(BaseTargetDriver):
    def __init__(self, featureset, target_spec: DataTargetSpec = None):
        self.name = "dataframe"
        self.featureset = featureset
        self.target = None

    def update_featureset_status(self, status="", producer=None):
        pass

    def add_writer_state(self, graph, after):
        key_column = self.featureset.spec.entities[0].name
        graph.add_step(
            name="WriteToDataFrame",
            after=after,
            shape="cylinder",
            class_name="storey.ReduceToDataFrame",
            index=key_column,
            insert_key_column_as=key_column,
            insert_time_column_as=self.featureset.spec.timestamp_key,
        )


kind_to_driver = {
    TargetTypes.parquet: ParquetTarget,
    TargetTypes.nosql: NoSqlTarget,
    TargetTypes.dataframe: DFTarget,
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


def _get_target_path(kind, featureset, suffix=""):
    name = featureset.metadata.name
    version = featureset.metadata.tag
    project = featureset.metadata.project or mlconf.default_project
    data_prefix = store_config.data_prefixes.get(kind, None)
    if not data_prefix:
        data_prefix = store_config.data_prefixes["default"]
    data_prefix = data_prefix.format(project=project, kind=kind)
    if version:
        name = f"{name}-{version}"
    return f"{data_prefix}/{name}{suffix}"
