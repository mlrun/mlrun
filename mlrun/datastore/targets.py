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
import random
import time
import typing
import warnings
from collections import Counter
from copy import copy
from typing import Union

import pandas as pd

import mlrun
import mlrun.utils.helpers
from mlrun.config import config
from mlrun.model import DataSource, DataTarget, DataTargetBase, TargetPathObject
from mlrun.utils import now_date
from mlrun.utils.v3io_clients import get_frames_client

from .. import errors
from ..data_types import ValueType
from ..platforms.iguazio import parse_path, split_path
from .utils import parse_kafka_url, store_path_to_spark


class TargetTypes:
    csv = "csv"
    parquet = "parquet"
    nosql = "nosql"
    redisnosql = "redisnosql"
    tsdb = "tsdb"
    stream = "stream"
    kafka = "kafka"
    dataframe = "dataframe"
    custom = "custom"

    @staticmethod
    def all():
        return [
            TargetTypes.csv,
            TargetTypes.parquet,
            TargetTypes.nosql,
            TargetTypes.redisnosql,
            TargetTypes.tsdb,
            TargetTypes.stream,
            TargetTypes.kafka,
            TargetTypes.dataframe,
            TargetTypes.custom,
        ]


def generate_target_run_id():
    return f"{round(time.time() * 1000)}_{random.randint(0, 999)}"


def default_target_names():
    targets = mlrun.mlconf.feature_store.default_targets
    return [target.strip() for target in targets.split(",")]


def get_default_targets():
    """initialize the default feature set targets list"""
    return [
        DataTargetBase(target, name=str(target), partitioned=(target == "parquet"))
        for target in default_target_names()
    ]


def update_targets_run_id_for_ingest(overwrite, targets, targets_in_status):
    run_id = generate_target_run_id()
    for target in targets:
        if overwrite or not (target.name in targets_in_status.keys()):
            target.run_id = run_id
        else:
            target.run_id = targets_in_status[target.name].run_id


def get_default_prefix_for_target(kind):
    data_prefixes = mlrun.mlconf.feature_store.data_prefixes
    data_prefix = getattr(data_prefixes, kind, None)
    if not data_prefix:
        data_prefix = data_prefixes.default
    return data_prefix


def get_default_prefix_for_source(kind):
    return get_default_prefix_for_target(kind)


def validate_target_paths_for_engine(
    targets, engine, source: Union[DataSource, pd.DataFrame]
):
    """Validating that target paths are suitable for the required engine.
    validate for single file targets only (parquet and csv).

    spark:
        cannot be a single file path (e.g - ends with .csv or .pq)

    storey:
        if csv - must be a single file.
        if parquet - in case of partitioned it must be a directory,
                     else can be both single file or directory

    pandas:
        if source contains chunksize attribute - path must be a directory
        else if parquet - if partitioned(=True) - path must be a directory
        else - path must be a single file


    :param targets:       list of data target objects
    :param engine:        name of the processing engine (storey, pandas, or spark), defaults to storey
    :param source:        source dataframe or other sources (e.g. parquet source see:
                          :py:class:`~mlrun.datastore.ParquetSource` and other classes in
                          mlrun.datastore with suffix Source)
    """
    for base_target in targets:
        if hasattr(base_target, "kind") and (
            base_target.kind == TargetTypes.parquet
            or base_target.kind == TargetTypes.csv
        ):
            target = get_target_driver(base_target)
            is_single_file = target.is_single_file()
            if engine == "spark" and is_single_file:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"spark CSV/Parquet targets must be directories, got path:'{target.path}'"
                )
            elif engine == "pandas":
                # check if source is DataSource (not DataFrame) and if contains chunk size
                if isinstance(source, DataSource) and source.attributes.get(
                    "chunksize"
                ):
                    if is_single_file:
                        raise mlrun.errors.MLRunInvalidArgumentError(
                            "pandas CSV/Parquet targets must be a directory "
                            f"for a chunked source, got path:'{target.path}'"
                        )
                elif target.kind == TargetTypes.parquet and target.partitioned:
                    if is_single_file:
                        raise mlrun.errors.MLRunInvalidArgumentError(
                            "partitioned Parquet target for pandas engine must be a directory, "
                            f"got path:'{target.path}'"
                        )
                elif not is_single_file:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "When using a non chunked source, "
                        f"pandas CSV/Parquet targets must be a single file, got path:'{target.path}'"
                    )
            elif not engine or engine == "storey":
                if target.kind == TargetTypes.csv and not is_single_file:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"CSV target for storey engine must be a single file, got path:'{target.path}'"
                    )
                elif (
                    target.kind == TargetTypes.parquet
                    and target.partitioned
                    and is_single_file
                ):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"partitioned Parquet target for storey engine must be a directory, got path:'{target.path}'"
                    )


def validate_target_list(targets):
    """Check that no target overrides another target in the list (name/path)"""

    if not targets:
        return
    targets_by_kind_name = [kind for kind in targets if type(kind) is str]
    no_name_target_types_count = Counter(
        [
            target.kind
            for target in targets
            if hasattr(target, "name") and hasattr(target, "kind") and not target.name
        ]
        + targets_by_kind_name
    )
    target_types_requiring_name = [
        target_type
        for target_type, target_type_count in no_name_target_types_count.items()
        if target_type_count > 1
    ]
    if target_types_requiring_name:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Only one default name per target type is allowed (please specify name for {0} target)".format(
                target_types_requiring_name
            )
        )

    target_names_count = Counter(
        [target.name for target in targets if hasattr(target, "name") and target.name]
    )

    targets_with_same_name = [
        target_name
        for target_name, target_name_count in target_names_count.items()
        if target_name_count > 1
    ]

    if targets_with_same_name:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Each target must have a unique name (more than one target with those names found {0})".format(
                targets_with_same_name
            )
        )

    no_path_target_types_count = Counter(
        [
            target.kind
            for target in targets
            if hasattr(target, "path") and hasattr(target, "kind") and not target.path
        ]
        + targets_by_kind_name
    )
    target_types_requiring_path = [
        target_type
        for target_type, target_type_count in no_path_target_types_count.items()
        if target_type_count > 1
    ]
    if target_types_requiring_path:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Only one default path per target type is allowed (please specify path for {0} target)".format(
                target_types_requiring_path
            )
        )

    target_paths_count = Counter(
        [target.path for target in targets if hasattr(target, "path") and target.path]
    )

    targets_with_same_path = [
        target_path
        for target_path, target_path_count in target_paths_count.items()
        if target_path_count > 1
    ]

    if targets_with_same_path:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Each target must have a unique path (more than one target with those names found {0})".format(
                targets_with_same_path
            )
        )


def validate_target_placement(graph, final_step, targets):
    if final_step or graph.is_empty():
        return True
    for target in targets:
        if not target.after_step:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "writer step location is undetermined due to graph branching"
                ", set the target .after_step attribute or the graph .final_step"
            )


def add_target_states(graph, resource, targets, to_df=False, final_state=None):
    warnings.warn(
        "This method is deprecated. Use add_target_steps instead",
        # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
        PendingDeprecationWarning,
    )
    return add_target_steps(graph, resource, targets, to_df, final_state)


def add_target_steps(graph, resource, targets, to_df=False, final_step=None):
    """add the target steps to the graph"""
    targets = targets or []
    key_columns = resource.spec.entities
    timestamp_key = resource.spec.timestamp_key
    features = resource.spec.features
    table = None

    for target in targets:
        driver = get_target_driver(target, resource)
        table = driver.get_table_object() or table
        driver.update_resource_status()
        driver.add_writer_step(
            graph,
            target.after_step or final_step,
            features=features if not target.after_step else None,
            key_columns=key_columns,
            timestamp_key=timestamp_key,
            featureset_status=resource.status,
        )
    if to_df:
        # add dataframe target, will return a dataframe
        driver = DFTarget()

        driver.add_writer_step(
            graph,
            final_step,
            features=features,
            key_columns=key_columns,
            timestamp_key=timestamp_key,
        )

    return table


offline_lookup_order = [TargetTypes.parquet, TargetTypes.csv]
online_lookup_order = [TargetTypes.nosql]


def get_offline_target(featureset, name=None):
    """return an optimal offline feature set target"""
    # todo: take status, start_time and lookup order into account
    offline_targets = [
        target
        for target in featureset.status.targets
        if kind_to_driver[target.kind].is_offline
    ]
    target = None
    if name:
        target = next((t for t in offline_targets if t.name == name), None)
    else:
        for kind in offline_lookup_order:
            target = next((t for t in offline_targets if t.kind == kind), None)
            if target:
                break
        if target is None and offline_targets:
            target = offline_targets[0]

    if target:
        return get_target_driver(target, featureset)
    return None


def get_online_target(resource, name=None):
    """return an optimal online feature set target"""
    # todo: take lookup order into account
    for target in resource.status.targets:
        if name and target.name != name:
            continue
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
    support_append = False

    def __init__(
        self,
        name: str = "",
        path=None,
        attributes: typing.Dict[str, str] = None,
        after_step=None,
        columns=None,
        partitioned: bool = False,
        key_bucketing_number: typing.Optional[int] = None,
        partition_cols: typing.Optional[typing.List[str]] = None,
        time_partitioning_granularity: typing.Optional[str] = None,
        after_state=None,
        max_events: typing.Optional[int] = None,
        flush_after_seconds: typing.Optional[int] = None,
        storage_options: typing.Dict[str, str] = None,
    ):
        super().__init__(
            self.kind,
            name,
            path,
            attributes,
            after_step,
            partitioned,
            key_bucketing_number,
            partition_cols,
            time_partitioning_granularity,
            max_events,
            flush_after_seconds,
            after_state,
        )
        if after_state:
            warnings.warn(
                "The after_state parameter is deprecated. Use after_step instead",
                # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
                PendingDeprecationWarning,
            )
            after_step = after_step or after_state

        self.name = name or self.kind
        self.path = str(path) if path is not None else None
        self.after_step = after_step
        self.attributes = attributes or {}
        self.columns = columns or []
        self.partitioned = partitioned
        self.key_bucketing_number = key_bucketing_number
        self.partition_cols = partition_cols
        self.time_partitioning_granularity = time_partitioning_granularity
        self.max_events = max_events
        self.flush_after_seconds = flush_after_seconds
        self.storage_options = storage_options

        self._target = None
        self._resource = None
        self._secrets = {}

    def _get_store(self):
        store, _ = mlrun.store_manager.get_or_create_store(self.get_target_path())
        return store

    def _get_column_list(self, features, timestamp_key, key_columns, with_type=False):
        result = []
        if self.columns:
            if with_type:
                columns = set(self.columns)
                for feature in features:
                    if feature.name in columns:
                        result.append((feature.name, feature.value_type))
            else:
                result = self.columns
        elif features:
            if with_type:
                for feature in features:
                    result.append((feature.name, feature.value_type))
            else:
                result = list(features.keys())
            if key_columns:
                for key in reversed(key_columns):
                    if key not in result:
                        if with_type:
                            result.insert(0, (key, ValueType.STRING))
                        else:
                            result.insert(0, key)

        if timestamp_key:
            if with_type:
                result = [(timestamp_key, ValueType.DATETIME)] + result
            else:
                result = [timestamp_key] + result

        return result

    def write_dataframe(
        self,
        df,
        key_column=None,
        timestamp_key=None,
        chunk_id=0,
        **kwargs,
    ) -> typing.Optional[int]:
        if hasattr(df, "rdd"):
            options = self.get_spark_options(key_column, timestamp_key)
            options.update(kwargs)
            df.write.mode("overwrite").save(**options)
        elif hasattr(df, "dask"):
            dask_options = self.get_dask_options()
            storage_options = self._get_store().get_storage_options()
            df = df.repartition(partition_size="100MB")
            try:
                if dask_options["format"] == "parquet":
                    df.to_parquet(
                        generate_path_with_chunk(self, chunk_id),
                        storage_options=storage_options,
                    )
                elif dask_options["format"] == "csv":
                    df.to_csv(
                        generate_path_with_chunk(self, chunk_id),
                        storage_options=storage_options,
                    )
                else:
                    raise NotImplementedError(
                        "Format for writing dask dataframe should be CSV or Parquet!"
                    )
            except Exception as exc:
                raise RuntimeError(f"Failed to write Dask Dataframe for {exc}.")
        else:
            target_path = generate_path_with_chunk(self, chunk_id)
            fs = self._get_store().get_filesystem(False)
            if fs.protocol == "file":
                dir = os.path.dirname(target_path)
                if dir:
                    os.makedirs(dir, exist_ok=True)
            partition_cols = []
            if target_path.endswith(".parquet") or target_path.endswith(".pq"):
                partition_cols = None
            target_df = df
            if timestamp_key and (
                self.partitioned or self.time_partitioning_granularity
            ):
                target_df = df.copy(deep=False)
                time_partitioning_granularity = self.time_partitioning_granularity
                if not time_partitioning_granularity and self.partitioned:
                    time_partitioning_granularity = (
                        mlrun.utils.helpers.DEFAULT_TIME_PARTITIONING_GRANULARITY
                    )
                for unit, fmt in [
                    ("year", "%Y"),
                    ("month", "%m"),
                    ("day", "%d"),
                    ("hour", "%H"),
                    ("minute", "%M"),
                ]:
                    partition_cols.append(unit)
                    target_df[unit] = getattr(
                        pd.DatetimeIndex(target_df[timestamp_key]), unit
                    )
                    if unit == time_partitioning_granularity:
                        break
            self._write_dataframe(
                target_df, fs, target_path, partition_cols=partition_cols, **kwargs
            )
            try:
                return fs.size(target_path)
            except Exception:
                return None

    @staticmethod
    def _write_dataframe(df, fs, target_path, partition_cols, **kwargs):
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

        if hasattr(spec, "columns"):
            driver.columns = spec.columns

        driver.partitioned = spec.partitioned

        driver.key_bucketing_number = spec.key_bucketing_number
        driver.partition_cols = spec.partition_cols

        driver.time_partitioning_granularity = spec.time_partitioning_granularity
        driver.max_events = spec.max_events
        driver.flush_after_seconds = spec.flush_after_seconds
        driver.storage_options = spec.storage_options

        driver._resource = resource
        driver.run_id = spec.run_id
        return driver

    def get_table_object(self):
        """get storey Table object"""
        return None

    def get_target_path(self):
        path_object = self._target_path_object
        return path_object.get_absolute_path() if path_object else None

    def get_target_templated_path(self):
        path_object = self._target_path_object
        return path_object.get_templated_path() if path_object else None

    @property
    def _target_path_object(self):
        """return the actual/computed target path"""
        is_single_file = hasattr(self, "is_single_file") and self.is_single_file()
        return self.get_path() or (
            TargetPathObject(
                _get_target_path(self, self._resource, self.run_id is not None),
                self.run_id,
                is_single_file,
            )
            if self._resource
            else None
        )

    def update_resource_status(self, status="", producer=None, size=None):
        """update the data target status"""
        self._target = self._target or DataTarget(
            self.kind, self.name, self.get_target_templated_path()
        )
        target = self._target
        target.run_id = self.run_id
        target.status = status or target.status or "created"
        target.updated = now_date().isoformat()
        target.size = size
        target.producer = producer or target.producer
        # Copy partitioning-related fields to the status, since these are needed if reading the actual data that
        # is related to the specific target.
        # TODO - instead of adding more fields to the status targets, we should consider changing any functionality
        #       that depends on "spec-fields" to use a merge between the status and the spec targets. One such place
        #       is the fset.to_dataframe() function.
        target.partitioned = self.partitioned
        target.key_bucketing_number = self.key_bucketing_number
        target.partition_cols = self.partition_cols
        target.time_partitioning_granularity = self.time_partitioning_granularity

        self._resource.status.update_target(target)
        return target

    def add_writer_step(
        self,
        graph,
        after,
        features,
        key_columns=None,
        timestamp_key=None,
        featureset_status=None,
    ):
        raise NotImplementedError()

    def add_writer_state(
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        warnings.warn(
            "This method is deprecated. Use add_writer_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        """add storey writer state to graph"""
        self.add_writer_step(graph, after, features, key_columns, timestamp_key)

    def purge(self):
        self._get_store().rm(self.get_target_path(), recursive=True)

    def as_df(
        self,
        columns=None,
        df_module=None,
        entities=None,
        start_time=None,
        end_time=None,
        time_column=None,
        **kwargs,
    ):
        """return the target data as dataframe"""
        return mlrun.get_dataitem(self.get_target_path()).as_df(
            columns=columns,
            df_module=df_module,
            start_time=start_time,
            end_time=end_time,
            time_column=time_column,
            **kwargs,
        )

    def get_spark_options(self, key_column=None, timestamp_key=None, overwrite=True):
        # options used in spark.read.load(**options)
        raise NotImplementedError()

    def prepare_spark_df(self, df):
        return df

    def get_dask_options(self):
        raise NotImplementedError()


class ParquetTarget(BaseStoreTarget):
    """parquet target storage driver, used to materialize feature set/vector data into parquet files

    :param name:       optional, target name. By default will be called ParquetTarget
    :param path:       optional, Output path. Can be either a file or directory.
     This parameter is forwarded as-is to pandas.DataFrame.to_parquet().
     Default location v3io:///projects/{project}/FeatureStore/{name}/parquet/
    :param attributes: optional, extra attributes for storey.ParquetTarget
    :param after_step: optional, after what step in the graph to add the target
    :param columns:     optional, which columns from data to write
    :param partitioned: optional, whether to partition the file, False by default,
     if True without passing any other partition field, the data will be partitioned by /year/month/day/hour
    :param key_bucketing_number:      optional, None by default will not partition by key,
     0 will partition by the key as is, any other number X will create X partitions and hash the keys to one of them
    :param partition_cols:     optional, name of columns from the data to partition by
    :param time_partitioning_granularity: optional. the smallest time unit to partition the data by.
     For example "hour" will yield partitions of the format /year/month/day/hour
    :param max_events: optional. Maximum number of events to write at a time.
     All events will be written on flow termination,
     or after flush_after_seconds (if flush_after_seconds is set). Default 10k events
    :param flush_after_seconds: optional. Maximum number of seconds to hold events before they are written.
     All events will be written on flow termination, or after max_events are accumulated (if max_events is set).
     Default 15 minutes
    """

    kind = TargetTypes.parquet
    is_offline = True
    support_spark = True
    support_storey = True
    support_dask = True
    support_append = True

    def __init__(
        self,
        name: str = "",
        path=None,
        attributes: typing.Dict[str, str] = None,
        after_step=None,
        columns=None,
        partitioned: bool = None,
        key_bucketing_number: typing.Optional[int] = None,
        partition_cols: typing.Optional[typing.List[str]] = None,
        time_partitioning_granularity: typing.Optional[str] = None,
        after_state=None,
        max_events: typing.Optional[int] = 10000,
        flush_after_seconds: typing.Optional[int] = 900,
        storage_options: typing.Dict[str, str] = None,
    ):
        if after_state:
            warnings.warn(
                "The after_state parameter is deprecated. Use after_step instead",
                # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
                PendingDeprecationWarning,
            )
            after_step = after_step or after_state

        self.path = path
        if partitioned is None:
            partitioned = not self.is_single_file()

        super().__init__(
            name,
            path,
            attributes,
            after_step,
            columns,
            partitioned,
            key_bucketing_number,
            partition_cols,
            time_partitioning_granularity,
            max_events=max_events,
            flush_after_seconds=flush_after_seconds,
            storage_options=storage_options,
        )

        if (
            time_partitioning_granularity is not None
            and time_partitioning_granularity
            not in mlrun.utils.helpers.LEGAL_TIME_UNITS
        ):
            raise errors.MLRunInvalidArgumentError(
                f"time_partitioning_granularity parameter must be one of "
                f"{','.join(mlrun.utils.helpers.LEGAL_TIME_UNITS)}, "
                f"not {time_partitioning_granularity}."
            )

    @staticmethod
    def _write_dataframe(df, fs, target_path, partition_cols, **kwargs):
        # In order to save the DataFrame in parquet format, all of the column names must be strings:
        df.columns = [str(column) for column in df.columns.tolist()]
        df.to_parquet(target_path, partition_cols=partition_cols, **kwargs)

    def add_writer_state(
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        warnings.warn(
            "This method is deprecated. Use add_writer_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        """add storey writer state to graph"""
        self.add_writer_step(graph, after, features, key_columns, timestamp_key)

    def add_writer_step(
        self,
        graph,
        after,
        features,
        key_columns=None,
        timestamp_key=None,
        featureset_status=None,
    ):
        if self.attributes.get("infer_columns_from_data"):
            column_list = None
        else:
            column_list = self._get_column_list(
                features=features,
                timestamp_key=timestamp_key,
                key_columns=None,
                with_type=True,
            )

        # need to extract types from features as part of column list

        partition_cols = None
        if self.key_bucketing_number is not None:
            partition_cols = [("$key", self.key_bucketing_number)]
        if self.partition_cols:
            partition_cols = partition_cols or []
            partition_cols.extend(self.partition_cols)
        time_partitioning_granularity = self.time_partitioning_granularity
        if self.partitioned and all(
            value is None
            for value in [
                time_partitioning_granularity,
                self.key_bucketing_number,
                self.partition_cols,
            ]
        ):
            time_partitioning_granularity = (
                mlrun.utils.helpers.DEFAULT_TIME_PARTITIONING_GRANULARITY
            )
        if time_partitioning_granularity is not None:
            partition_cols = partition_cols or []
            for time_unit in mlrun.utils.helpers.LEGAL_TIME_UNITS:
                partition_cols.append(f"${time_unit}")
                if time_unit == time_partitioning_granularity:
                    break

        if (
            not self.partitioned
            and not self.get_target_path().endswith(".parquet")
            and not self.get_target_path().endswith(".pq")
        ):
            partition_cols = []

        tuple_key_columns = []
        for key_column in key_columns:
            tuple_key_columns.append((key_column.name, key_column.value_type))

        if self.attributes:
            self.attributes[
                "update_last_written"
            ] = featureset_status.update_last_written_for_target
        else:
            self.attributes = {
                "update_last_written": featureset_status.update_last_written_for_target
            }

        graph.add_step(
            name=self.name or "ParquetTarget",
            after=after,
            graph_shape="cylinder",
            class_name="storey.ParquetTarget",
            path=self.get_target_path(),
            columns=column_list,
            index_cols=tuple_key_columns,
            partition_cols=partition_cols,
            storage_options=self.storage_options
            or self._get_store().get_storage_options(),
            max_events=self.max_events,
            flush_after_seconds=self.flush_after_seconds,
            **self.attributes,
        )

    def get_spark_options(self, key_column=None, timestamp_key=None, overwrite=True):
        partition_cols = []
        if timestamp_key:
            time_partitioning_granularity = self.time_partitioning_granularity
            if (
                not time_partitioning_granularity
                and self.partitioned
                and not self.partition_cols
            ):
                time_partitioning_granularity = (
                    mlrun.utils.helpers.DEFAULT_TIME_PARTITIONING_GRANULARITY
                )
            if time_partitioning_granularity:
                for unit in mlrun.utils.helpers.LEGAL_TIME_UNITS:
                    partition_cols.append(unit)
                    if unit == time_partitioning_granularity:
                        break
        result = {
            "path": store_path_to_spark(self.get_target_path()),
            "format": "parquet",
        }
        for partition_col in self.partition_cols or []:
            partition_cols.append(partition_col)
        if partition_cols:
            result["partitionBy"] = partition_cols
        return result

    def get_dask_options(self):
        return {"format": "parquet"}

    def as_df(
        self,
        columns=None,
        df_module=None,
        entities=None,
        start_time=None,
        end_time=None,
        time_column=None,
        **kwargs,
    ):
        """return the target data as dataframe"""
        return mlrun.get_dataitem(self.get_target_path()).as_df(
            columns=columns,
            df_module=df_module,
            format="parquet",
            start_time=start_time,
            end_time=end_time,
            time_column=time_column,
            **kwargs,
        )

    def is_single_file(self):
        if self.path:
            return self.path.endswith(".parquet") or self.path.endswith(".pq")
        return False


class CSVTarget(BaseStoreTarget):
    kind = TargetTypes.csv
    suffix = ".csv"
    is_offline = True
    support_spark = True
    support_storey = True

    @staticmethod
    def _write_dataframe(df, fs, target_path, partition_cols, **kwargs):
        with fs.open(target_path, "wb") as fp:
            # avoid writing the range index unless explicitly specified via kwargs
            if isinstance(df.index, pd.RangeIndex):
                kwargs["index"] = kwargs.get("index", False)
            df.to_csv(fp, **kwargs)

    def add_writer_state(
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        warnings.warn(
            "This method is deprecated. Use add_writer_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        """add storey writer state to graph"""
        self.add_writer_step(graph, after, features, key_columns, timestamp_key)

    def add_writer_step(
        self,
        graph,
        after,
        features,
        key_columns=None,
        timestamp_key=None,
        featureset_status=None,
    ):
        key_columns = list(key_columns.keys())
        column_list = self._get_column_list(
            features=features, timestamp_key=timestamp_key, key_columns=key_columns
        )

        graph.add_step(
            name=self.name or "CSVTarget",
            after=after,
            graph_shape="cylinder",
            class_name="storey.CSVTarget",
            path=self.get_target_path(),
            columns=column_list,
            header=True,
            index_cols=key_columns,
            storage_options=self._get_store().get_storage_options(),
            **self.attributes,
        )

    def get_spark_options(self, key_column=None, timestamp_key=None, overwrite=True):
        return {
            "path": store_path_to_spark(self.get_target_path()),
            "format": "csv",
            "header": "true",
        }

    def prepare_spark_df(self, df):
        import pyspark.sql.functions as funcs

        for col_name, col_type in df.dtypes:
            if col_type == "timestamp":
                # df.write.csv saves timestamps with millisecond precision, but we want microsecond precision
                # for compatibility with storey.
                df = df.withColumn(
                    col_name, funcs.date_format(col_name, "yyyy-MM-dd HH:mm:ss.SSSSSS")
                )
        return df

    def as_df(
        self,
        columns=None,
        df_module=None,
        entities=None,
        start_time=None,
        end_time=None,
        time_column=None,
        **kwargs,
    ):
        df = super().as_df(
            columns=columns, df_module=df_module, entities=entities, **kwargs
        )
        df.set_index(keys=entities, inplace=True)
        return df

    def is_single_file(self):
        if self.path:
            return self.path.endswith(".csv")
        return True


class NoSqlBaseTarget(BaseStoreTarget):
    is_table = True
    is_online = True
    support_append = True
    support_storey = True
    writer_step_name = "base_name"

    def __new__(cls, *args, **kwargs):
        if cls is NoSqlBaseTarget:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return object.__new__(cls)

    def get_table_object(self):
        raise NotImplementedError()

    def add_writer_state(
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        warnings.warn(
            "This method is deprecated. Use add_writer_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        """add storey writer state to graph"""
        self.add_writer_step(graph, after, features, key_columns, timestamp_key)

    def add_writer_step(
        self,
        graph,
        after,
        features,
        key_columns=None,
        timestamp_key=None,
        featureset_status=None,
    ):
        key_columns = list(key_columns.keys())
        table = self._resource.uri
        column_list = self._get_column_list(
            features=features,
            timestamp_key=None,
            key_columns=key_columns,
            with_type=True,
        )
        if not self.columns:
            aggregate_features = (
                [key for key, feature in features.items() if feature.aggregate]
                if features
                else []
            )
            column_list = [
                col for col in column_list if col[0] not in aggregate_features
            ]

        graph.add_step(
            name=self.name or self.writer_step_name,
            after=after,
            graph_shape="cylinder",
            class_name="storey.NoSqlTarget",
            columns=column_list,
            table=table,
            **self.attributes,
        )

    def get_spark_options(self, key_column=None, timestamp_key=None, overwrite=True):
        spark_options = {
            "path": store_path_to_spark(self.get_target_path()),
            "format": "io.iguaz.v3io.spark.sql.kv",
        }
        if isinstance(key_column, list) and len(key_column) >= 1:
            if len(key_column) > 2:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Spark supports maximun of 2 keys and {key_column} are provided"
                )
            spark_options["key"] = key_column[0]
            if len(key_column) > 1:
                spark_options["sorting-key"] = key_column[1]
        else:
            spark_options["key"] = key_column
        if not overwrite:
            spark_options["columnUpdate"] = True
        return spark_options

    def get_dask_options(self):
        return {"format": "csv"}

    def as_df(self, columns=None, df_module=None, **kwargs):
        raise NotImplementedError()

    def prepare_spark_df(self, df):
        import pyspark.sql.functions as funcs

        for col_name, col_type in df.dtypes:
            if col_type.startswith("decimal("):
                # V3IO does not support this level of precision
                df = df.withColumn(col_name, funcs.col(col_name).cast("double"))
        return df

    def write_dataframe(
        self, df, key_column=None, timestamp_key=None, chunk_id=0, **kwargs
    ):
        if hasattr(df, "rdd"):
            options = self.get_spark_options(key_column, timestamp_key)
            options.update(kwargs)
            df = self.prepare_spark_df(df)
            df.write.mode("overwrite").save(**options)
        else:
            access_key = self._secrets.get(
                "V3IO_ACCESS_KEY", os.getenv("V3IO_ACCESS_KEY")
            )

            _, path_with_container = parse_path(self.get_target_path())
            container, path = split_path(path_with_container)

            frames_client = get_frames_client(
                token=access_key, address=config.v3io_framesd, container=container
            )

            frames_client.write("kv", path, df, index_cols=key_column, **kwargs)


class NoSqlTarget(NoSqlBaseTarget):
    kind = TargetTypes.nosql
    support_spark = True
    writer_step_name = "NoSqlTarget"

    def get_table_object(self):
        from storey import Table, V3ioDriver

        # TODO use options/cred
        endpoint, uri = parse_path(self.get_target_path())
        return Table(
            uri,
            V3ioDriver(webapi=endpoint),
            flush_interval_secs=mlrun.mlconf.feature_store.flush_interval,
        )


class RedisNoSqlTarget(NoSqlBaseTarget):
    kind = TargetTypes.redisnosql
    support_spark = False
    writer_step_name = "RedisNoSqlTarget"

    def get_table_object(self):
        from storey import Table
        from storey.redis_driver import RedisDriver

        endpoint, uri = parse_path(self.get_target_path())
        endpoint = endpoint or mlrun.mlconf.redis.url

        return Table(
            uri,
            RedisDriver(redis_url=endpoint, key_prefix="/"),
            flush_interval_secs=mlrun.mlconf.feature_store.flush_interval,
        )


class StreamTarget(BaseStoreTarget):
    kind = TargetTypes.stream
    is_table = False
    is_online = False
    support_spark = False
    support_storey = True
    support_append = True

    def add_writer_state(
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        warnings.warn(
            "This method is deprecated. Use add_writer_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        """add storey writer state to graph"""
        self.add_writer_step(graph, after, features, key_columns, timestamp_key)

    def add_writer_step(
        self,
        graph,
        after,
        features,
        key_columns=None,
        timestamp_key=None,
        featureset_status=None,
    ):
        from storey import V3ioDriver

        key_columns = list(key_columns.keys())
        endpoint, uri = parse_path(self.get_target_path())
        column_list = self._get_column_list(
            features=features, timestamp_key=timestamp_key, key_columns=key_columns
        )

        graph.add_step(
            name=self.name or "StreamTarget",
            after=after,
            graph_shape="cylinder",
            class_name="storey.StreamTarget",
            columns=column_list,
            storage=V3ioDriver(webapi=endpoint),
            stream_path=uri,
            **self.attributes,
        )

    def as_df(self, columns=None, df_module=None, **kwargs):
        raise NotImplementedError()


class KafkaTarget(BaseStoreTarget):
    kind = TargetTypes.kafka
    is_table = False
    is_online = False
    support_spark = False
    support_storey = True
    support_append = True

    def __init__(
        self,
        *args,
        bootstrap_servers=None,
        producer_options=None,
        **kwargs,
    ):
        attrs = {
            "bootstrap_servers": bootstrap_servers,
            "producer_options": producer_options,
        }
        super().__init__(*args, attributes=attrs, **kwargs)

    def add_writer_step(
        self,
        graph,
        after,
        features,
        key_columns=None,
        timestamp_key=None,
        featureset_status=None,
    ):
        key_columns = list(key_columns.keys())
        column_list = self._get_column_list(
            features=features, timestamp_key=timestamp_key, key_columns=key_columns
        )

        bootstrap_servers = self.attributes.get("bootstrap_servers")
        topic, bootstrap_servers = parse_kafka_url(self.path, bootstrap_servers)

        graph.add_step(
            name=self.name or "KafkaTarget",
            after=after,
            graph_shape="cylinder",
            class_name="storey.KafkaTarget",
            columns=column_list,
            topic=topic,
            bootstrap_servers=bootstrap_servers,
            **self.attributes,
        )

    def as_df(self, columns=None, df_module=None, **kwargs):
        raise NotImplementedError()


class TSDBTarget(BaseStoreTarget):
    kind = TargetTypes.tsdb
    is_table = False
    is_online = False
    support_spark = False
    support_storey = True
    support_append = True

    def add_writer_state(
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        warnings.warn(
            "This method is deprecated. Use add_writer_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        """add storey writer state to graph"""
        self.add_writer_step(graph, after, features, key_columns, timestamp_key)

    def add_writer_step(
        self,
        graph,
        after,
        features,
        key_columns=None,
        timestamp_key=None,
        featureset_status=None,
    ):
        key_columns = list(key_columns.keys())
        endpoint, uri = parse_path(self.get_target_path())
        if not timestamp_key:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "feature set timestamp_key must be specified for TSDBTarget writer"
            )

        column_list = self._get_column_list(
            features=features, timestamp_key=None, key_columns=key_columns
        )

        graph.add_step(
            name=self.name or "TSDBTarget",
            class_name="storey.TSDBTarget",
            after=after,
            graph_shape="cylinder",
            path=uri,
            time_col=timestamp_key,
            index_cols=key_columns,
            columns=column_list,
            **self.attributes,
        )

    def as_df(self, columns=None, df_module=None, **kwargs):
        raise NotImplementedError()

    def write_dataframe(
        self, df, key_column=None, timestamp_key=None, chunk_id=0, **kwargs
    ):
        access_key = self._secrets.get("V3IO_ACCESS_KEY", os.getenv("V3IO_ACCESS_KEY"))

        new_index = []
        if timestamp_key:
            new_index.append(timestamp_key)
        if key_column:
            if isinstance(key_column, str):
                key_column = [key_column]
            new_index.extend(key_column)

        _, path_with_container = parse_path(self.get_target_path())
        container, path = split_path(path_with_container)

        frames_client = get_frames_client(
            token=access_key,
            address=config.v3io_framesd,
            container=container,
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
        self,
        class_name: str,
        name: str = "",
        after_step=None,
        after_state=None,
        **attributes,
    ):
        if after_state:
            warnings.warn(
                "The after_state parameter is deprecated. Use after_step instead",
                # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
                PendingDeprecationWarning,
            )
            after_step = after_step or after_state

        attributes = attributes or {}
        attributes["class_name"] = class_name
        super().__init__(name, "", attributes, after_step=after_step)

    def add_writer_state(
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        warnings.warn(
            "This method is deprecated. Use add_writer_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        """add storey writer state to graph"""
        self.add_writer_step(graph, after, features, key_columns, timestamp_key)

    def add_writer_step(
        self,
        graph,
        after,
        features,
        key_columns=None,
        timestamp_key=None,
        featureset_status=None,
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
    kind = TargetTypes.dataframe
    support_storey = True

    def __init__(self, *args, name="dataframe", **kwargs):
        self._df = None
        super().__init__(*args, name=name, **kwargs)

    def set_df(self, df):
        self._df = df

    def update_resource_status(self, status="", producer=None):
        pass

    def add_writer_state(
        self, graph, after, features, key_columns=None, timestamp_key=None
    ):
        warnings.warn(
            "This method is deprecated. Use add_writer_step instead",
            # TODO: In 0.7.0 do changes in examples & demos In 0.9.0 remove
            PendingDeprecationWarning,
        )
        """add storey writer state to graph"""
        self.add_writer_step(graph, after, features, key_columns, timestamp_key)

    def add_writer_step(
        self,
        graph,
        after,
        features,
        key_columns=None,
        timestamp_key=None,
        featureset_status=None,
    ):
        key_columns = list(key_columns.keys())
        # todo: column filter
        graph.add_step(
            name=self.name or "WriteToDataFrame",
            after=after,
            graph_shape="cylinder",
            class_name="storey.ReduceToDataFrame",
            index=key_columns,
            insert_key_column_as=key_columns,
            insert_time_column_as=timestamp_key,
        )

    def as_df(
        self,
        columns=None,
        df_module=None,
        start_time=None,
        end_time=None,
        time_column=None,
        **kwargs,
    ):
        return self._df


kind_to_driver = {
    TargetTypes.parquet: ParquetTarget,
    TargetTypes.csv: CSVTarget,
    TargetTypes.nosql: NoSqlTarget,
    TargetTypes.redisnosql: RedisNoSqlTarget,
    TargetTypes.dataframe: DFTarget,
    TargetTypes.stream: StreamTarget,
    TargetTypes.kafka: KafkaTarget,
    TargetTypes.tsdb: TSDBTarget,
    TargetTypes.custom: CustomTarget,
}


def _get_target_path(driver, resource, run_id_mode=False):
    """return the default target path given the resource and target kind"""
    kind = driver.kind
    suffix = driver.suffix
    if not suffix:
        if (
            kind == ParquetTarget.kind
            and resource.kind == mlrun.api.schemas.ObjectKind.feature_vector
        ):
            suffix = ".parquet"
    kind_prefix = (
        "sets"
        if resource.kind == mlrun.api.schemas.ObjectKind.feature_set
        else "vectors"
    )
    name = resource.metadata.name
    project = resource.metadata.project or mlrun.mlconf.default_project
    data_prefix = get_default_prefix_for_target(kind).format(
        project=project,
        kind=kind,
        name=name,
    )
    # todo: handle ver tag changes, may need to copy files?
    if not run_id_mode:
        version = resource.metadata.tag
        name = f"{name}-{version or 'latest'}"
    return f"{data_prefix}/{kind_prefix}/{name}{suffix}"


def generate_path_with_chunk(target, chunk_id):
    prefix, suffix = os.path.splitext(target.get_target_path())
    if chunk_id and not target.partitioned and not target.time_partitioning_granularity:
        return f"{prefix}/{chunk_id:0>4}{suffix}"
    return target.get_target_path()
