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
#
import numpy as np
import pandas as pd
import pyarrow
from pandas.io.json._table_schema import convert_pandas_type_to_json_field

from .data_types import InferOptions, pa_type_to_value_type, pd_schema_to_value_type

default_num_bins = 20


def infer_schema_from_df(
    df: pd.DataFrame,
    features,
    entities,
    timestamp_key: str = None,
    entity_columns=None,
    options: InferOptions = InferOptions.Null,
):
    """infer feature set schema from dataframe"""
    timestamp_fields = []
    current_entities = list(entities.keys())
    entity_columns = entity_columns or []
    index_columns = dict()

    def upsert_entity(name, value_type):
        if name in current_entities:
            entities[name].value_type = value_type
        else:
            entities[name] = {"name": name, "value_type": value_type}

    # remove index column if no name provided
    if not df.index.name and df.index.is_numeric():
        df = df.reset_index().drop("index", axis=1)

    schema = pyarrow.Schema.from_pandas(df)
    index_type = None
    for i in range(len(schema)):
        column = schema.names[i]
        value_type = pa_type_to_value_type(schema.types[i])
        if column in df.index.names:
            index_columns[column] = value_type
            continue
        is_entity = column in entity_columns or column in current_entities
        if is_entity:
            upsert_entity(column, value_type)
        elif (
            InferOptions.get_common_options(options, InferOptions.Features)
            and column != timestamp_key
        ):
            if column in features.keys():
                features[column].value_type = value_type
            else:
                features[column] = {"name": column, "value_type": value_type}
        if value_type == "datetime" and not is_entity:
            timestamp_fields.append(column)

    index_type = None
    if InferOptions.get_common_options(options, InferOptions.Index):
        # infer types of index fields
        if df.index.name:
            if df.index.name in index_columns:
                index_type = index_columns[df.index.name]
            if not index_type:
                field = convert_pandas_type_to_json_field(df.index)
                index_type = pd_schema_to_value_type(field["type"])
            # Workaround to infer a boolean index correctly, and not as 'str'.
            upsert_entity(df.index.name, index_type)
        elif df.index.nlevels > 1:
            for level, name in zip(df.index.levels, df.index.names):
                if name in index_columns:
                    index_type = index_columns[name]
                else:
                    field = convert_pandas_type_to_json_field(df.index)
                    index_type = pd_schema_to_value_type(field["type"])
                upsert_entity(name, index_type)
                if index_type == "datetime":
                    timestamp_fields.append(name)

    return timestamp_key


def get_df_stats(df, options, num_bins=None, sample_size=None):
    """get per column data stats from dataframe"""

    results_dict = {}
    if df.empty:
        return results_dict
    if sample_size and df.shape[0] > sample_size:
        df = df.sample(sample_size)

    num_bins = num_bins or default_num_bins
    if InferOptions.get_common_options(options, InferOptions.Index) and df.index.names:
        df = df.reset_index()
    for col, values in df.describe(include="all", datetime_is_numeric=True).items():
        stats_dict = {}
        for stat, val in values.dropna().items():
            if isinstance(val, (float, np.floating, np.float64)):
                stats_dict[stat] = float(val)
            elif isinstance(val, (int, np.integer, np.int64)):
                # boolean values are considered subclass of int
                if isinstance(val, bool):
                    stats_dict[stat] = bool(val)
                else:
                    stats_dict[stat] = int(val)
            else:
                stats_dict[stat] = str(val)

        if InferOptions.get_common_options(
            options, InferOptions.Histogram
        ) and pd.api.types.is_numeric_dtype(df[col]):
            # store histogram
            try:
                hist, bins = np.histogram(df[col], bins=num_bins)
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
        shortdf = df.sample(preview_lines)
    shortdf = shortdf.reset_index(inplace=False)
    return [shortdf.columns.values.tolist()] + shortdf.values.tolist()


class DFDataInfer:
    infer_schema = infer_schema_from_df
    get_preview = get_df_preview
    get_stats = get_df_stats
