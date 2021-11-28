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

    values = [df.select(funcs.collect_list(val)).first()[0] for val in df.columns]
    preview = [df.columns]
    for row in list(zip(*values)):
        preview.append(list(row))
    return preview


def _create_hist_data(df, column, minim, maxim, bins=10):
    def create_all_conditions(current_col, column, left_edges, count=1):
        """
        Recursive function that exploits the
        ability to call the Spark SQL Column method
        .when() in a recursive way.
        """
        left_edges = left_edges[:]
        if len(left_edges) == 0:
            return current_col
        if len(left_edges) == 1:
            next_col = current_col.when(
                funcs.col(column) >= float(left_edges[0]), count
            )
            left_edges.pop(0)
            return create_all_conditions(next_col, column, left_edges[:], count + 1)
        next_col = current_col.when(
            (float(left_edges[0]) <= funcs.col(column))
            & (funcs.col(column) < float(left_edges[1])),
            count,
        )
        left_edges.pop(0)
        return create_all_conditions(next_col, column, left_edges[:], count + 1)

    num_range = maxim - minim
    bin_width = num_range / float(bins)
    left_edges = [minim]

    for _bin in range(bins):
        left_edges = left_edges + [left_edges[-1] + bin_width]
    left_edges.pop()
    expression_col = funcs.when(
        (float(left_edges[0]) <= funcs.col(column))
        & (funcs.col(column) < float(left_edges[1])),
        0,
    )
    left_edges_copy = left_edges[:]
    left_edges_copy.pop(0)
    bin_data = (
        df.select(funcs.col(column))
        .na.drop()
        .select(
            funcs.col(column),
            create_all_conditions(expression_col, column, left_edges_copy).alias(
                "bin_id"
            ),
        )
        .groupBy("bin_id")
        .count()
    ).toPandas()

    bin_data.index = bin_data["bin_id"]
    new_index = list(range(bins))
    bin_data = bin_data.reindex(new_index)
    bin_data["bin_id"] = bin_data.index
    bin_data = bin_data.fillna(0)
    bin_data["left_edge"] = left_edges
    bin_data["width"] = bin_width
    bin_data = [
        bin_data["count"].tolist(),
        [round(x, 2) for x in bin_data["left_edge"].tolist()],
    ]

    return bin_data


def get_dtype(df, colname):
    return [dtype for name, dtype in df.dtypes if name == colname][0]


def get_df_stats_spark(df, options, num_bins=20, sample_size=None):
    if InferOptions.get_common_options(options, InferOptions.Index):
        df = df.select("*").withColumn("id", funcs.monotonically_increasing_id())

    # todo: sample spark DF if sample_size is not None and DF is bigger than sample_size

    summary_df = df.summary().toPandas()
    summary_df.set_index(["summary"], drop=True, inplace=True)
    results_dict = {}
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

        if InferOptions.get_common_options(
            options, InferOptions.Histogram
        ) and get_dtype(df, col) in ["double", "int"]:
            try:
                results_dict[col]["hist"] = _create_hist_data(
                    df,
                    col,
                    float(results_dict[col]["min"]),
                    float(results_dict[col]["max"]),
                    bins=num_bins,
                )
            except Exception:
                pass

    return results_dict


class SparkDataInfer:
    infer_schema = infer_schema_from_df_spark
    get_preview = get_df_preview_spark
    get_stats = get_df_stats_spark
