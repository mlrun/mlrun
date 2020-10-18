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

import numpy as np
import pandas as pd
from pandas.io.json._table_schema import convert_pandas_type_to_json_field

from .model import Feature, FeatureSetSpec, FeatureSetStatus, Entity
from .datatypes import pd_schema_to_value_type


def infer_features_from_df(
    df: pd.DataFrame,
    featureset_spec: FeatureSetSpec,
    entity_columns=None,
    with_index=True,
):
    """infer feature set schema from dataframe"""
    features = []
    entities = []
    timestamp_fields = []

    for column, s in df.items():
        value_type = _get_column_type(s)
        is_entity = entity_columns and column in entity_columns
        if is_entity:
            entities.append(Entity(name=column, value_type=value_type))
        else:
            features.append(Feature(name=column, value_type=value_type))
        if value_type == "datetime":
            timestamp_fields.append((column, is_entity))

    if with_index:
        # infer types of index fields
        if df.index.name:
            value_type = _get_column_type(df.index)
            entities.append(Entity(name=df.index.name, value_type=value_type))
            if value_type == "datetime":
                timestamp_fields.append((df.index.name, True))
        elif df.index.nlevels > 1:
            for level, name in zip(df.index.levels, df.index.names):
                value_type = _get_column_type(level)
                entities.append(Entity(name=name, value_type=value_type))
                if value_type == "datetime":
                    timestamp_fields.append((name, True))

    featureset_spec.features = features
    featureset_spec.entities = entities
    if len(timestamp_fields) == 1:
        featureset_spec.timestamp_key = timestamp_fields[0][0]


def _get_column_type(column):
    field = convert_pandas_type_to_json_field(column)
    return pd_schema_to_value_type(field["type"])


def get_df_stats(
    df,
    featureset_status: FeatureSetStatus,
    with_index=True,
    with_histogram=False,
    with_preview=False,
    preview_lines=20,
):
    """get per column data stats from dataframe"""

    d = {}
    if with_index:
        df = df.reset_index()
    for col, values in df.describe(include="all", percentiles=[]).items():
        stats_dict = {}
        for stat, val in values.dropna().items():
            if stat != "50%":
                if isinstance(val, (float, np.floating, np.float64)):
                    stats_dict[stat] = float(val)
                elif isinstance(val, (int, np.integer, np.int64)):
                    stats_dict[stat] = int(val)
                else:
                    stats_dict[stat] = str(val)

        if with_histogram and pd.api.types.is_numeric_dtype(df[col]):
            # store histogram
            try:
                hist, bins = np.histogram(df[col], bins=20)
                stats_dict["hist"] = [hist.tolist(), bins.tolist()]
            except Exception:
                pass

        d[col] = stats_dict
    featureset_status.stats = d

    if with_preview:
        # record sample rows from the dataframe
        length = df.shape[0]
        shortdf = df
        if length > preview_lines:
            shortdf = df.head(preview_lines)
        featureset_status.preview = [
            shortdf.columns.values.tolist()
        ] + shortdf.values.tolist()

    return d
