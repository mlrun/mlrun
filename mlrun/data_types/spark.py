from os import environ

import numpy as np

from .data_types import InferOptions, spark_to_value_type

try:
    import pyspark.sql.functions as funcs
except ImportError:
    pass


def infer_schema_from_df_spark(
    df,
    features,
    entities,
    timestamp_key: str = None,
    entity_columns=None,
    options: InferOptions = InferOptions.Null,
):
    timestamp_fields = []
    current_entities = list(entities.keys())
    entity_columns = entity_columns or []

    def upsert_entity(name, value_type):
        if name in current_entities:
            entities[name].value_type = value_type
        else:
            entities[name] = {"name": name, "value_type": value_type}

    for column, s in df.dtypes:
        value_type = spark_to_value_type(s)
        is_entity = column in entity_columns or column in current_entities
        if is_entity:
            upsert_entity(column, value_type)
        elif options & InferOptions.Features and column != timestamp_key:
            if column in features.keys():
                features[column].value_type = value_type
            else:
                features[column] = {"name": column, "value_type": value_type}
        if value_type == "timestamp" and not is_entity:
            timestamp_fields.append(column)

    return timestamp_key


def get_df_preview_spark(df, preview_lines=20):
    """capture preview data from spark df"""
    df = df.limit(preview_lines)

    result_dict = df.toPandas().to_dict(orient="split")
    return [result_dict["columns"], *result_dict["data"]]


def _create_hist_data(df, calculation_cols, results_dict, bins=20):
    hist_df = df
    col_names = []
    bins_values = {}

    for orig_col in calculation_cols:
        bins_start = np.linspace(
            float(results_dict[orig_col]["min"]),
            float(results_dict[orig_col]["max"]),
            bins,
            endpoint=False,
        ).tolist()
        bins_values[orig_col] = bins_start

        for i in range(bins):
            col_name = f"{orig_col}_hist_bin_{i}"
            if i < (bins - 1):
                hist_df = hist_df.withColumn(
                    col_name,
                    funcs.when(
                        (funcs.col(orig_col) >= bins_start[i])
                        & (funcs.col(orig_col) < bins_start[i + 1]),
                        1,
                    ).otherwise(0),
                )
            else:
                hist_df = hist_df.withColumn(
                    col_name,
                    funcs.when(funcs.col(orig_col) >= bins_start[i], 1).otherwise(0),
                )
            col_names.append(col_name)

    agg_funcs = [funcs.sum(hist_df[col]).alias(col) for col in col_names]
    hist_values = hist_df.groupBy().agg(*agg_funcs).toPandas()

    for orig_col in calculation_cols:
        bin_data = [
            [hist_values.loc[0][f"{orig_col}_hist_bin_{i}"] for i in range(bins)],
            [round(x, 2) for x in bins_values[orig_col]],
        ]
        results_dict[orig_col]["hist"] = bin_data


def get_dtype(df, colname):
    return [dtype for name, dtype in df.dtypes if name == colname][0]


# Histogram calculation in Spark relies on adding a column to the DF per histogram bin. Spark supports many
# columns, but not infinitely (and there's no hard-limit on number of columns). This value will determine
# how many histograms will be calculated in a single query. By default we're using 20 bins per histogram, so
# using 500 will calculate histograms over 25 columns in a single query. If there are more, more queries will
# be executed.
MAX_HISTOGRAM_COLUMNS_IN_QUERY = int(
    environ.get("MLRUN_MAX_HISTOGRAM_COLUMNS_IN_QUERY", 500)
)


def get_df_stats_spark(df, options, num_bins=20, sample_size=None):
    if InferOptions.get_common_options(options, InferOptions.Index):
        df = df.select("*").withColumn("id", funcs.monotonically_increasing_id())

    # todo: sample spark DF if sample_size is not None and DF is bigger than sample_size

    summary_df = df.summary().toPandas()
    summary_df.set_index(["summary"], drop=True, inplace=True)
    results_dict = {}
    hist_columns = []
    for col, values in summary_df.items():
        stats_dict = {}
        for stat, val in values.dropna().items():
            if stat != "50%":
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
        results_dict[col] = stats_dict

        if (
            InferOptions.get_common_options(options, InferOptions.Histogram)
            and get_dtype(df, col) in ["double", "int"]
            # in some situations (empty DF may cause this) we won't have stats for columns with suitable types, in this
            # case we won't calculate histogram for the column.
            and "min" in results_dict[col]
            and "max" in results_dict[col]
        ):
            hist_columns.append(col)

    # We may need multiple queries here. See comment before this function for reasoning.
    max_columns_per_query = int(MAX_HISTOGRAM_COLUMNS_IN_QUERY // num_bins) or 1
    while len(hist_columns) > 0:
        calculation_cols = hist_columns[:max_columns_per_query]
        _create_hist_data(df, calculation_cols, results_dict, num_bins)
        hist_columns = hist_columns[max_columns_per_query:]

    return results_dict


class SparkDataInfer:
    infer_schema = infer_schema_from_df_spark
    get_preview = get_df_preview_spark
    get_stats = get_df_stats_spark
