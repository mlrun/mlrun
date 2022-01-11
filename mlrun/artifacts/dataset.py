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
import pathlib
import typing
from io import StringIO

import numpy as np
import pandas as pd
from pandas.io.json import build_table_schema

import mlrun
import mlrun.utils.helpers

from ..datastore import is_store_uri, store_manager
from .base import Artifact

default_preview_rows_length = 20
max_preview_columns = 100
max_csv = 10000
ddf_sample_pct = 0.2
max_ddf_size = 1


class TableArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + ["schema", "header"]
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
            self.header = df.reset_index().columns.values.tolist()
            self.format = "csv"  # todo other formats
            # if visible and not key_suffix:
            #     key += '.csv'
            self._body = df
        else:
            self._is_df = False
            self.header = header

        self.schema = schema
        if not viewer:
            viewer = "table" if visible else None
        self.viewer = viewer

    def get_body(self):
        if not self._is_df:
            return self._body
        csv_buffer = StringIO()
        self._body.to_csv(csv_buffer, line_terminator="\n", encoding="utf-8")
        return csv_buffer.getvalue()


supported_formats = ["csv", "parquet", "pq", "tsdb", "kv"]


class DatasetArtifact(Artifact):
    _dict_fields = Artifact._dict_fields + [
        "schema",
        "header",
        "length",
        "preview",
        "stats",
        "extra_data",
        "column_metadata",
    ]
    kind = "dataset"

    def __init__(
        self,
        key=None,
        df=None,
        preview=None,
        format="",
        stats=None,
        target_path=None,
        extra_data=None,
        column_metadata=None,
        ignore_preview_limits=False,
        **kwargs,
    ):

        format = format.lower()
        super().__init__(key, None, format=format, target_path=target_path)
        if format and format not in supported_formats:
            raise ValueError(
                f"unsupported format {format} use one of {'|'.join(supported_formats)}"
            )

        if format == "pq":
            format = "parquet"
        self.format = format
        self.stats = None
        self.extra_data = extra_data or {}
        self.column_metadata = column_metadata or {}

        if df is not None:
            if hasattr(df, "dask"):
                # If df is a Dask DataFrame, and it's small in-memory, convert to Pandas
                if (df.memory_usage(deep=True).sum().compute() / 1e9) < max_ddf_size:
                    df = df.compute()
            self.update_preview_fields_from_df(
                self, df, stats, preview, ignore_preview_limits
            )

        self._df = df
        self._kw = kwargs

    def upload(self):
        suffix = pathlib.Path(self.target_path).suffix
        format = self.format
        if not format:
            if suffix and suffix in [".csv", ".parquet", ".pq"]:
                format = "csv" if suffix == ".csv" else "parquet"
            else:
                format = "parquet"
        if not suffix and not self.target_path.startswith("memory://"):
            self.target_path = self.target_path + "." + format

        self.size, self.hash = upload_dataframe(
            self._df,
            self.target_path,
            format=format,
            src_path=self.src_path,
            **self._kw,
        )

    @staticmethod
    def update_preview_fields_from_df(
        artifact, df, stats=None, preview_rows_length=None, ignore_preview_limits=False
    ):
        preview_rows_length = preview_rows_length or default_preview_rows_length
        if hasattr(df, "dask"):
            artifact.length = df.shape[0].compute()
            preview_df = df.sample(frac=ddf_sample_pct).compute()
        else:
            artifact.length = df.shape[0]
            preview_df = df

        if artifact.length > preview_rows_length and not ignore_preview_limits:
            preview_df = df.head(preview_rows_length)
        preview_df = preview_df.reset_index()
        if len(preview_df.columns) > max_preview_columns and not ignore_preview_limits:
            preview_df = preview_df.iloc[:, :max_preview_columns]
        artifact.header = preview_df.columns.values.tolist()
        artifact.preview = preview_df.values.tolist()
        artifact.schema = build_table_schema(preview_df)
        if (
            stats
            or (artifact.length < max_csv and len(df.columns) < max_preview_columns)
            or ignore_preview_limits
        ):
            artifact.stats = get_df_stats(df)


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
    schema: dict = None,
    header: list = None,
    preview: list = None,
    stats: dict = None,
    extra_data: dict = None,
    column_metadata: dict = None,
    labels: dict = None,
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
    elif is_store_uri(artifact):
        artifact_spec, _ = store_manager.get_store_artifact(artifact)
    else:
        raise ValueError("model path must be a model store object/URL/DataItem")

    if not artifact_spec or artifact_spec.kind != "dataset":
        raise ValueError(f"store artifact ({artifact}) is not dataset kind")

    if from_df is not None:
        DatasetArtifact.update_preview_fields_from_df(
            artifact_spec, from_df, stats, ignore_preview_limits
        )

    if header:
        artifact_spec.header = header
    if stats:
        artifact_spec.stats = stats
    if schema:
        artifact_spec.schema = schema
    if preview:
        artifact_spec.preview = preview
    if column_metadata:
        artifact_spec.column_metadata = column_metadata
    if labels:
        for key, val in labels.items():
            artifact_spec.labels[key] = val

    if extra_data:
        artifact_spec.extra_data = artifact_spec.extra_data or {}
        for key, item in extra_data.items():
            if hasattr(item, "target_path"):
                item = item.target_path
            artifact_spec.extra_data[key] = item

    mlrun.get_run_db().store_artifact(
        artifact_spec.db_key,
        artifact_spec.to_dict(),
        artifact_spec.tree,
        iter=artifact_spec.iter,
        project=artifact_spec.project,
    )


def upload_dataframe(
    df, target_path, format, src_path=None, **kw
) -> typing.Tuple[typing.Optional[int], typing.Optional[str]]:
    if src_path and os.path.isfile(src_path):
        store_manager.object(url=target_path).upload(src_path)
        return (
            os.stat(src_path).st_size,
            mlrun.utils.helpers.calculate_local_file_hash(src_path),
        )

    if df is None:
        return None, None

    if target_path.startswith("memory://"):
        store_manager.object(target_path).put(df)
        return None, None

    if format in ["csv", "parquet"]:
        target_class = mlrun.datastore.targets.kind_to_driver[format]
        size = target_class(path=target_path).write_dataframe(df, **kw)
        return size, None

    raise mlrun.errors.MLRunInvalidArgumentError(f"format {format} not implemented yes")
