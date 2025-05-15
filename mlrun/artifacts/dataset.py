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
import os
import pathlib
import warnings
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
from pandas.io.json import build_table_schema

import mlrun
import mlrun.common.schemas
import mlrun.datastore
import mlrun.utils.helpers
from mlrun.config import config as mlconf

from .base import Artifact, ArtifactSpec, StorePrefix

default_preview_rows_length = 20
max_preview_columns = mlconf.artifacts.datasets.max_preview_columns
max_csv = 10000
ddf_sample_pct = 0.2
max_ddf_size = 1


class TableArtifactSpec(ArtifactSpec):
    _dict_fields = ArtifactSpec._dict_fields + ["schema", "header"]
    _exclude_fields_from_uid_hash = ArtifactSpec._exclude_fields_from_uid_hash + [
        "schema",
        "header",
    ]

    def __init__(self):
        super().__init__()
        self.schema = None
        self.header = None


class TableArtifact(Artifact):
    kind = "table"

    def __init__(
        self,
        key=None,
        body=None,
        df=None,
        viewer=None,
        visible=False,
        inline=False,
        format=None,
        header=None,
        schema=None,
    ):
        if key:
            key_suffix = pathlib.Path(key).suffix
            if not format and key_suffix:
                format = key_suffix[1:]
        super().__init__(key, body, viewer=viewer, is_inline=inline, format=format)

        if df is not None:
            self._is_df = True
            self.spec.header = df.reset_index(drop=True).columns.values.tolist()
            self.spec.format = "csv"  # todo other formats
            # if visible and not key_suffix:
            #     key += '.csv'
            self.spec._body = df
        else:
            self._is_df = False
            self.spec.header = header

        self.spec.schema = schema
        if not viewer:
            viewer = "table" if visible else None
        self.spec.viewer = viewer

    @property
    def spec(self) -> TableArtifactSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", TableArtifactSpec)

    def get_body(self):
        if not self._is_df:
            return self.spec.get_body()
        csv_buffer = StringIO()
        self.spec.get_body().to_csv(
            csv_buffer,
            encoding="utf-8",
            **mlrun.utils.line_terminator_kwargs(),
        )
        return csv_buffer.getvalue()


class DatasetArtifactSpec(ArtifactSpec):
    _dict_fields = ArtifactSpec._dict_fields + [
        "schema",
        "header",
        "length",
        "column_metadata",
        "features",
        "partition_keys",
        "timestamp_key",
        "label_column",
    ]

    _exclude_fields_from_uid_hash = ArtifactSpec._exclude_fields_from_uid_hash + [
        "schema",
        "header",
        "length",
        "column_metadata",
        "features",
        "partition_keys",
        "timestamp_key",
        "label_column",
    ]

    def __init__(self):
        super().__init__()
        self.schema = None
        self.header = None
        self.length = None
        self.column_metadata = None
        self.features = None
        self.partition_keys = None
        self.timestamp_key = None
        self.label_column = None


class DatasetArtifact(Artifact):
    kind = mlrun.common.schemas.ArtifactCategories.dataset
    # List of all the supported saving formats of a DataFrame:
    SUPPORTED_FORMATS = ["csv", "parquet", "pq", "tsdb", "kv"]
    _store_prefix = StorePrefix.Dataset

    def __init__(
        self,
        key: Optional[str] = None,
        df=None,
        preview: Optional[int] = None,
        format: str = "",  # TODO: should be changed to 'fmt'.
        stats: Optional[bool] = None,
        target_path: Optional[str] = None,
        extra_data: Optional[dict] = None,
        column_metadata: Optional[dict] = None,
        ignore_preview_limits: bool = False,
        label_column: Optional[str] = None,
        **kwargs,
    ):
        if key or format or target_path:
            warnings.warn(
                "Artifact constructor parameters are deprecated in 1.7.0 and will be removed in 1.10.0. "
                "Use the metadata and spec parameters instead.",
                DeprecationWarning,
            )

        format = (format or "").lower()
        super().__init__(key, None, format=format, target_path=target_path)
        if format and format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format {format} use one of {'|'.join(self.SUPPORTED_FORMATS)}"
            )

        if format == "pq":
            format = "parquet"
        self.format = format
        self.status.stats = None
        self.extra_data = extra_data or {}
        self.column_metadata = column_metadata or {}
        self.spec.label_column = label_column

        if df is not None:
            if label_column and label_column not in df.columns:
                raise mlrun.errors.MLRunValueError(
                    f"Provided dataframe doesn't include a column \"{label_column}\", so it can't be used as label"
                )
            if hasattr(df, "dask"):
                # If df is a Dask DataFrame, and it's small in-memory, convert to Pandas
                if (df.memory_usage(deep=True).sum().compute() / 1e9) < max_ddf_size:
                    df = df.compute()
            self.update_preview_fields_from_df(
                self, df, stats, preview, ignore_preview_limits
            )

        self._df = df
        self._kw = kwargs

    @property
    def spec(self) -> DatasetArtifactSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", DatasetArtifactSpec)

    def upload(self, artifact_path: Optional[str] = None):
        """
        internal, upload to target store
        :param artifact_path: required only for when generating target_path from artifact hash
        """
        if not self.spec.target_path:
            if self.spec.src_path:
                (
                    self.metadata.hash,
                    self.spec.target_path,
                ) = self.resolve_file_target_hash_path(
                    self.spec.src_path, artifact_path=artifact_path
                )
            else:
                (
                    self.metadata.hash,
                    self.spec.target_path,
                ) = self.resolve_dataframe_target_hash_path(
                    self._df, artifact_path=artifact_path
                )

        suffix = pathlib.Path(self.spec.target_path).suffix
        format = self.spec.format
        if not format:
            if suffix and suffix in [".csv", ".parquet", ".pq"]:
                format = "csv" if suffix == ".csv" else "parquet"
            else:
                format = "parquet"
        if not suffix and not self.spec.target_path.startswith("memory://"):
            self.spec.target_path = self.spec.target_path + "." + format

        if self._df is not None:
            self.spec.size, self.metadata.hash = upload_dataframe(
                self._df,
                self.spec.target_path,
                format=format,
                src_path=self.spec.src_path,
                **self._kw,
            )
        else:
            body = self.get_body()
            if body:
                self._upload_body(
                    body=body, target=self.target_path, artifact_path=artifact_path
                )
            else:
                # don't fail if no df or body
                self.spec.size, self.metadata.hash = None, None

    def resolve_dataframe_target_hash_path(self, dataframe, artifact_path: str):
        if not artifact_path:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Unable to resolve body target hash path, artifact_path is not defined"
            )
        dataframe_hash = mlrun.utils.helpers.calculate_dataframe_hash(dataframe)
        suffix = self._resolve_suffix()
        artifact_path = (
            artifact_path + "/" if not artifact_path.endswith("/") else artifact_path
        )
        target_path = f"{artifact_path}{dataframe_hash}{suffix}"
        return dataframe_hash, target_path

    @property
    def df(self) -> pd.DataFrame:
        """
        Get the dataset in this artifact.

        :return: The dataset as a DataFrame.
        """
        return self._df

    @staticmethod
    def is_format_supported(fmt: str) -> bool:
        """
        Check whether the given dataset format is supported by the DatasetArtifact.

        :param fmt: The format string to check.

        :return: True if the format is supported and False if not.
        """
        return fmt in DatasetArtifact.SUPPORTED_FORMATS

    @staticmethod
    def update_preview_fields_from_df(
        artifact, df, stats=None, preview_rows_length=None, ignore_preview_limits=False
    ):
        preview_rows_length = preview_rows_length or default_preview_rows_length
        if hasattr(df, "dask"):
            artifact.spec.length = df.shape[0].compute()
            preview_df = df.sample(frac=ddf_sample_pct).compute()
        else:
            artifact.spec.length = df.shape[0]
            preview_df = df

        if artifact.spec.length > preview_rows_length and not ignore_preview_limits:
            preview_df = df.head(preview_rows_length)

        preview_df = preview_df.reset_index()
        artifact.status.header_original_length = len(preview_df.columns)
        if len(preview_df.columns) > max_preview_columns and not ignore_preview_limits:
            preview_df = preview_df.iloc[:, :max_preview_columns]
        artifact.spec.header = preview_df.columns.values.tolist()
        artifact.status.preview = preview_df.values.tolist()
        # Table schema parsing doesn't require a column named "index"
        # to align its output with previously generated header and preview data
        if "index" in preview_df.columns:
            preview_df.drop("index", axis=1, inplace=True)
        artifact.spec.schema = build_table_schema(preview_df)

        # set artifact stats if stats is explicitly set to true, or if stats is None and the dataframe is small
        if stats or (
            stats is None
            and (
                artifact.spec.length < max_csv and len(df.columns) < max_preview_columns
            )
            or ignore_preview_limits
        ):
            artifact.status.stats = get_df_stats(df)

    @property
    def column_metadata(self):
        return self.spec.column_metadata

    @column_metadata.setter
    def column_metadata(self, column_metadata):
        self.spec.column_metadata = column_metadata

    @property
    def schema(self):
        return self.spec.schema

    @schema.setter
    def schema(self, schema):
        self.spec.schema = schema

    @property
    def header(self):
        return self.spec.header

    @header.setter
    def header(self, header):
        self.spec.header = header

    @property
    def preview(self):
        return self.status.preview

    @preview.setter
    def preview(self, preview):
        self.status.preview = preview

    @property
    def stats(self):
        return self.status.stats

    @stats.setter
    def stats(self, stats):
        self.status.stats = stats


def get_df_stats(df):
    if hasattr(df, "dask"):
        df = df.sample(frac=ddf_sample_pct).compute()
    d = {}
    for col, values in df.describe(include="all").items():
        stats_dict = {}
        for stat, val in values.dropna().items():
            if isinstance(val, (float, np.floating, np.float64)):
                stats_dict[stat] = float(val)
            elif isinstance(val, (int, np.integer, np.int64)):
                stats_dict[stat] = int(val)
            else:
                stats_dict[stat] = str(val)

        if pd.api.types.is_numeric_dtype(df[col]):
            # store histogram
            try:
                hist, bins = np.histogram(df[col], bins=20)
                stats_dict["hist"] = [hist.tolist(), bins.tolist()]
            except Exception:
                pass

        d[col] = stats_dict
    return d


def update_dataset_meta(
    artifact,
    from_df=None,
    schema: Optional[dict] = None,
    header: Optional[list] = None,
    preview: Optional[list] = None,
    stats: Optional[dict] = None,
    extra_data: Optional[dict] = None,
    column_metadata: Optional[dict] = None,
    labels: Optional[dict] = None,
    ignore_preview_limits: bool = False,
):
    """Update dataset object attributes/metadata

    this method will edit or add metadata to a dataset object

    example:
        update_dataset_meta(dataset, from_df=df,
                            extra_data={'histogram': 's3://mybucket/..'})

    :param from_df:                 read metadata (schema, preview, ..) from provided df
    :param artifact:                dataset artifact object or path (store://..) or DataItem
    :param schema:                  dataset schema, see pandas build_table_schema
    :param header:                  column headers
    :param preview:                 list of rows and row values (from df.values.tolist())
    :param stats:                   dict of column names and their stats (cleaned df.describe(include='all'))
    :param extra_data:              extra data items (key: path string | artifact)
    :param column_metadata:         dict of metadata per column
    :param labels:                  metadata labels
    :param ignore_preview_limits:   whether to ignore the preview size limits
    """

    if hasattr(artifact, "artifact_url"):
        artifact = artifact.artifact_url

    if isinstance(artifact, DatasetArtifact):
        artifact_spec = artifact
    elif mlrun.datastore.is_store_uri(artifact):
        artifact_spec, _ = mlrun.datastore.store_manager.get_store_artifact(artifact)
    else:
        raise ValueError("model path must be a model store object/URL/DataItem")

    if not artifact_spec or artifact_spec.kind != "dataset":
        raise ValueError(f"store artifact ({artifact}) is not dataset kind")

    if from_df is not None:
        DatasetArtifact.update_preview_fields_from_df(
            artifact_spec, from_df, stats, ignore_preview_limits
        )

    if header:
        artifact_spec.spec.header = header
    if stats:
        artifact_spec.status.stats = stats
    if schema:
        artifact_spec.spec.schema = schema
    if preview:
        artifact_spec.status.preview = preview
    if column_metadata:
        artifact_spec.spec.column_metadata = column_metadata
    if labels:
        for key, val in labels.items():
            artifact_spec.metadata.labels[key] = val

    if extra_data:
        artifact_spec.spec.extra_data = artifact_spec.spec.extra_data or {}
        for key, item in extra_data.items():
            if hasattr(item, "target_path"):
                item = item.spec.target_path
            artifact_spec.spec.extra_data[key] = item

    mlrun.get_run_db().store_artifact(
        artifact_spec.spec.db_key,
        artifact_spec.to_dict(),
        tree=artifact_spec.metadata.tree,
        iter=artifact_spec.metadata.iter,
        project=artifact_spec.metadata.project,
    )


def upload_dataframe(
    df, target_path, format, src_path=None, **kw
) -> tuple[Optional[int], Optional[str]]:
    if src_path and os.path.isfile(src_path):
        mlrun.datastore.store_manager.object(url=target_path).upload(src_path)
        return (
            os.stat(src_path).st_size,
            mlrun.utils.helpers.calculate_local_file_hash(src_path),
        )

    if df is None:
        return None, None

    if target_path.startswith("memory://"):
        mlrun.datastore.store_manager.object(target_path).put(df)
        return None, None

    if format in ["csv", "parquet"]:
        target_class = mlrun.datastore.targets.kind_to_driver[format]
        size = target_class(path=target_path).write_dataframe(df, **kw)
        return size, None

    raise mlrun.errors.MLRunInvalidArgumentError(f"Format {format} not implemented yet")
