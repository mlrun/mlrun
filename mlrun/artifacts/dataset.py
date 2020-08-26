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
from io import StringIO
from tempfile import mktemp
import numpy as np
import pandas as pd

from pandas.io.json import build_table_schema

from .base import Artifact
from ..datastore import store_manager
from ..utils import DB_SCHEMA

preview_lines = 20
max_csv = 10000


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
        **kwargs,
    ):

        format = format.lower()
        super().__init__(key, None, format=format, target_path=target_path)
        if format and format not in supported_formats:
            raise ValueError(
                "unsupported format {} use one of {}".format(
                    format, "|".join(supported_formats)
                )
            )

        if format == "pq":
            format = "parquet"
        self.format = format
        self.stats = None
        self.extra_data = extra_data or {}
        self.column_metadata = column_metadata or {}

        if df is not None:
            self.length = df.shape[0]
            preview = preview or preview_lines
            shortdf = df
            if self.length > preview:
                shortdf = df.head(preview)
            shortdf = shortdf.reset_index()
            self.header = shortdf.columns.values.tolist()
            self.preview = shortdf.values.tolist()
            self.schema = build_table_schema(df)
            if stats or self.length < max_csv:
                self.stats = get_df_stats(df)

        self._df = df
        self._kw = kwargs

    def upload(self, data_stores):
        suffix = pathlib.Path(self.target_path).suffix
        if not self.format:
            if suffix and suffix in [".csv", ".parquet", ".pq"]:
                self.format = "csv" if suffix == ".csv" else "parquet"
            else:
                self.format = "csv" if self.length < max_csv else "parquet"

        src_path = self.src_path
        if src_path and os.path.isfile(src_path):
            self._upload_file(src_path, data_stores)
            return

        if self._df is None:
            return

        if self.target_path.startswith("memory://"):
            data_stores.object(self.target_path).put(self._df)
            return

        if self.format in ["csv", "parquet"]:
            if not suffix:
                self.target_path = self.target_path + "." + self.format
            writer_string = "to_{}".format(self.format)
            saving_func = getattr(self._df, writer_string, None)
            target = self.target_path
            to_upload = False
            if "://" in target:
                target = mktemp()
                to_upload = True
            else:
                dir = os.path.dirname(target)
                if dir:
                    os.makedirs(dir, exist_ok=True)

            saving_func(target, **self._kw)
            if to_upload:
                self._upload_file(target, data_stores)
                self._set_meta(target)
                os.remove(target)
            else:
                self._set_meta(target)
            return

        raise ValueError(f"format {self.format} not implemented yes")


def get_df_stats(df):
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
):
    """Update dataset object attributes/metadata

    this method will edit or add metadata to a dataset object

    example:
        update_dataset_meta(dataset, from_df=df,
                            extra_data={'histogram': 's3://mybucket/..'})

    :param from_df:         read metadata (schema, preview, ..) from provided df
    :param artifact:        dataset artifact object or path (store://..) or DataItem
    :param schema:          dataset schema, see pandas build_table_schema
    :param header:          column headers
    :param preview:         list of rows and row values (from df.values.tolist())
    :param stats:           dict of column names and their stats (cleaned df.describe(include='all'))
    :param extra_data:      extra data items (key: path string | artifact)
    :param column_metadata: dict of metadata per column
    :param labels:          metadata labels
    """

    if hasattr(artifact, "artifact_url"):
        artifact = artifact.artifact_url

    stores = store_manager
    if isinstance(artifact, DatasetArtifact):
        artifact_spec = artifact
    elif artifact.startswith(DB_SCHEMA + "://"):
        artifact_spec, _ = stores.get_store_artifact(artifact)
    else:
        raise ValueError("model path must be a model store object/URL/DataItem")

    if not artifact_spec or artifact_spec.kind != "dataset":
        raise ValueError("store artifact ({}) is not dataset kind".format(artifact))

    if from_df is not None:
        shortdf = from_df
        length = from_df.shape[0]
        if length > preview_lines:
            shortdf = from_df.head(preview_lines)
        artifact_spec.header = shortdf.reset_index().columns.values.tolist()
        artifact_spec.preview = shortdf.reset_index().values.tolist()
        artifact_spec.schema = build_table_schema(from_df)
        if stats is None and length < max_csv:
            artifact_spec.stats = get_df_stats(from_df)

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

    stores._get_db().store_artifact(
        artifact_spec.db_key,
        artifact_spec.to_dict(),
        artifact_spec.tree,
        iter=artifact_spec.iter,
        project=artifact_spec.project,
    )
