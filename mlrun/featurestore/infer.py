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


def infer_schema_from_df(
    df: pd.DataFrame,
    featureset_spec: FeatureSetSpec,
    entity_columns=None,
    with_index=True,
    with_features=True,
):
    """infer feature set schema from dataframe"""
    features = []
    timestamp_fields = []
    current_entities = list(featureset_spec.entities.keys())
    entity_columns = entity_columns or []

    for column, s in df.items():
        value_type = _get_column_type(s)
        is_entity = column in entity_columns or column in current_entities
        if is_entity:
            if column not in current_entities:
                featureset_spec.entities[column] = Entity(value_type=value_type)
        elif with_features and column != featureset_spec.timestamp_key:
            features.append(Feature(name=column, value_type=value_type))
        if value_type == "datetime" and not is_entity:
            timestamp_fields.append(column)

    if with_index:
        # infer types of index fields
        if df.index.name:
            if column not in current_entities:
                value_type = _get_column_type(df.index)
                featureset_spec.entities[df.index.name] = Entity(value_type=value_type)
        elif df.index.nlevels > 1:
            for level, name in zip(df.index.levels, df.index.names):
                if column not in current_entities:
                    value_type = _get_column_type(level)
                    featureset_spec.entities[name] = Entity(value_type=value_type)
                    if value_type == "datetime":
                        timestamp_fields.append(name)

    if with_features:
        featureset_spec.features = features
    if len(timestamp_fields) == 1 and not featureset_spec.timestamp_key:
        featureset_spec.timestamp_key = timestamp_fields[0]


def _get_column_type(column):
    field = convert_pandas_type_to_json_field(column)
    return pd_schema_to_value_type(field["type"])


def get_df_stats(
    df,
    with_index=True,
    with_histogram=False,
):
    """get per column data stats from dataframe"""

    results_dict = {}
    if with_index:
        df = df.reset_index()
    for col, values in df.describe(
        include="all", percentiles=[], datetime_is_numeric=True
    ).items():
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

        results_dict[col] = stats_dict
    return results_dict


def get_df_preview(df, preview_lines=20):
    """capture preview data from df"""
    # record sample rows from the dataframe
    length = df.shape[0]
    shortdf = df
    if length > preview_lines:
        shortdf = df.head(preview_lines)
    return [shortdf.columns.values.tolist()] + shortdf.values.tolist()
