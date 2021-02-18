import pandas as pd
import numpy as np

from .data_types import InferOptions, pd_schema_to_value_type
from pandas.io.json._table_schema import convert_pandas_type_to_json_field

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

    def upsert_entity(name, value_type):
        if name in current_entities:
            entities[name].value_type = value_type
        else:
            entities[name] = {"name": name, "value_type": value_type}

    for column, series in df.items():
        value_type = _get_column_type(series)
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

    if InferOptions.get_common_options(options, InferOptions.Index):
        # infer types of index fields
        if df.index.name:
            value_type = _get_column_type(df.index)
            upsert_entity(df.index.name, value_type)
        elif df.index.nlevels > 1:
            for level, name in zip(df.index.levels, df.index.names):
                value_type = _get_column_type(level)
                upsert_entity(name, value_type)
                if value_type == "datetime":
                    timestamp_fields.append(name)

    if len(timestamp_fields) == 1 and not timestamp_key:
        return timestamp_fields[0]
    return timestamp_key


def _get_column_type(column):
    field = convert_pandas_type_to_json_field(column)
    return pd_schema_to_value_type(field["type"])


def get_df_stats(df, options, num_bins=None):
    """get per column data stats from dataframe"""

    results_dict = {}
    num_bins = num_bins or default_num_bins
    if InferOptions.get_common_options(options, InferOptions.Index) and df.index.name:
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
        shortdf = df.head(preview_lines)
    return [shortdf.columns.values.tolist()] + shortdf.values.tolist()


class DFDataInfer:
    infer_schema = infer_schema_from_df
    get_preview = get_df_preview
    get_stats = get_df_stats
