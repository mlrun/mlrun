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

import warnings
from collections import Counter

import pandas as pd
import semver


def _to_pandas(spark_df):
    """
    Modified version of spark DataFrame.toPandas() -
    https://github.com/apache/spark/blob/v3.2.3/python/pyspark/sql/pandas/conversion.py#L35

    The original code (which is only replaced in pyspark 3.5.0) fails with Pandas 2 installed, with the following error:
    Casting to unit-less dtype 'datetime64' is not supported. Pass e.g. 'datetime64[ns]' instead.

    This modification adds the missing unit to the dtype.
    """
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql.types import (
        BooleanType,
        IntegralType,
        MapType,
        TimestampType,
    )

    assert isinstance(spark_df, DataFrame)

    from pyspark.sql.pandas.utils import require_minimum_pandas_version

    require_minimum_pandas_version()

    import numpy as np

    timezone = spark_df.sql_ctx._conf.sessionLocalTimeZone()

    if spark_df.sql_ctx._conf.arrowPySparkEnabled():
        use_arrow = True
        try:
            from pyspark.sql.pandas.types import to_arrow_schema
            from pyspark.sql.pandas.utils import require_minimum_pyarrow_version

            require_minimum_pyarrow_version()
            to_arrow_schema(spark_df.schema)
        except Exception as e:
            if spark_df.sql_ctx._conf.arrowPySparkFallbackEnabled():
                msg = (
                    "toPandas attempted Arrow optimization because "
                    "'spark.sql.execution.arrow.pyspark.enabled' is set to true; however, "
                    f"failed by the reason below:\n  {e}\n"
                    "Attempting non-optimization as "
                    "'spark.sql.execution.arrow.pyspark.fallback.enabled' is set to "
                    "true."
                )
                warnings.warn(msg)
                use_arrow = False
            else:
                msg = (
                    "toPandas attempted Arrow optimization because "
                    "'spark.sql.execution.arrow.pyspark.enabled' is set to true, but has "
                    "reached the error below and will not continue because automatic fallback "
                    "with 'spark.sql.execution.arrow.pyspark.fallback.enabled' has been set to "
                    f"false.\n  {e}"
                )
                warnings.warn(msg)
                raise

        # Try to use Arrow optimization when the schema is supported and the required version
        # of PyArrow is found, if 'spark.sql.execution.arrow.pyspark.enabled' is enabled.
        if use_arrow:
            try:
                import pyarrow
                from pyspark.sql.pandas.types import (
                    _check_series_localize_timestamps,
                    _convert_map_items_to_dict,
                )

                # Rename columns to avoid duplicated column names.
                tmp_column_names = [f"col_{i}" for i in range(len(spark_df.columns))]
                self_destruct = spark_df.sql_ctx._conf.arrowPySparkSelfDestructEnabled()
                batches = spark_df.toDF(*tmp_column_names)._collect_as_arrow(
                    split_batches=self_destruct
                )
                if len(batches) > 0:
                    table = pyarrow.Table.from_batches(batches)
                    # Ensure only the table has a reference to the batches, so that
                    # self_destruct (if enabled) is effective
                    del batches
                    # Pandas DataFrame created from PyArrow uses datetime64[ns] for date type
                    # values, but we should use datetime.date to match the behavior with when
                    # Arrow optimization is disabled.
                    pandas_options = {"date_as_object": True}
                    if self_destruct:
                        # Configure PyArrow to use as little memory as possible:
                        # self_destruct - free columns as they are converted
                        # split_blocks - create a separate Pandas block for each column
                        # use_threads - convert one column at a time
                        pandas_options.update(
                            {
                                "self_destruct": True,
                                "split_blocks": True,
                                "use_threads": False,
                            }
                        )
                    pdf = table.to_pandas(**pandas_options)
                    # Rename back to the original column names.
                    pdf.columns = spark_df.columns
                    for field in spark_df.schema:
                        if isinstance(field.dataType, TimestampType):
                            pdf[field.name] = _check_series_localize_timestamps(
                                pdf[field.name], timezone
                            )
                        elif isinstance(field.dataType, MapType):
                            pdf[field.name] = _convert_map_items_to_dict(
                                pdf[field.name]
                            )
                    return pdf
                else:
                    return pd.DataFrame.from_records([], columns=spark_df.columns)
            except Exception as e:
                # We might have to allow fallback here as well but multiple Spark jobs can
                # be executed. So, simply fail in this case for now.
                msg = (
                    "toPandas attempted Arrow optimization because "
                    "'spark.sql.execution.arrow.pyspark.enabled' is set to true, but has "
                    "reached the error below and can not continue. Note that "
                    "'spark.sql.execution.arrow.pyspark.fallback.enabled' does not have an "
                    "effect on failures in the middle of "
                    f"computation.\n  {e}"
                )
                warnings.warn(msg)
                raise

    # Below is toPandas without Arrow optimization.
    pdf = pd.DataFrame.from_records(spark_df.collect(), columns=spark_df.columns)
    column_counter = Counter(spark_df.columns)

    dtype = [None] * len(spark_df.schema)
    for field_idx, field in enumerate(spark_df.schema):
        # For duplicate column name, we use `iloc` to access it.
        if column_counter[field.name] > 1:
            pandas_col = pdf.iloc[:, field_idx]
        else:
            pandas_col = pdf[field.name]

        pandas_type = _to_corrected_pandas_type(field.dataType)
        # SPARK-21766: if an integer field is nullable and has null values, it can be
        # inferred by pandas as float column. Once we convert the column with NaN back
        # to integer type e.g., np.int16, we will hit exception. So we use the inferred
        # float type, not the corrected type from the schema in this case.
        if pandas_type is not None and not (
            isinstance(field.dataType, IntegralType)
            and field.nullable
            and pandas_col.isnull().any()
        ):
            dtype[field_idx] = pandas_type
        # Ensure we fall back to nullable numpy types, even when whole column is null:
        if isinstance(field.dataType, IntegralType) and pandas_col.isnull().any():
            dtype[field_idx] = np.float64
        if isinstance(field.dataType, BooleanType) and pandas_col.isnull().any():
            dtype[field_idx] = object

    df = pd.DataFrame()
    for index, t in enumerate(dtype):
        column_name = spark_df.schema[index].name

        # For duplicate column name, we use `iloc` to access it.
        if column_counter[column_name] > 1:
            series = pdf.iloc[:, index]
        else:
            series = pdf[column_name]

        if t is not None:
            series = series.astype(t, copy=False)

        # `insert` API makes copy of data, we only do it for Series of duplicate column names.
        # `pdf.iloc[:, index] = pdf.iloc[:, index]...` doesn't always work because `iloc` could
        # return a view or a copy depending by context.
        if column_counter[column_name] > 1:
            df.insert(index, column_name, series, allow_duplicates=True)
        else:
            df[column_name] = series

    pdf = df

    if timezone is None:
        return pdf
    else:
        from pyspark.sql.pandas.types import _check_series_convert_timestamps_local_tz

        for field in spark_df.schema:
            # TODO: handle nested timestamps, such as ArrayType(TimestampType())?
            if isinstance(field.dataType, TimestampType):
                pdf[field.name] = _check_series_convert_timestamps_local_tz(
                    pdf[field.name], timezone
                )
        return pdf


def _to_corrected_pandas_type(dt):
    import numpy as np
    from pyspark.sql.types import (
        BooleanType,
        ByteType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        TimestampType,
    )

    if isinstance(dt, ByteType):
        return np.int8
    elif isinstance(dt, ShortType):
        return np.int16
    elif isinstance(dt, IntegerType):
        return np.int32
    elif isinstance(dt, LongType):
        return np.int64
    elif isinstance(dt, FloatType):
        return np.float32
    elif isinstance(dt, DoubleType):
        return np.float64
    elif isinstance(dt, BooleanType):
        return bool
    elif isinstance(dt, TimestampType):
        return "datetime64[ns]"
    else:
        return None


def spark_df_to_pandas(spark_df):
    import pyspark

    if semver.parse(pyspark.__version__) >= semver.Version(3, 5, 0):

        def to_pandas(spark_df_inner):
            return spark_df_inner.toPandas()
    else:
        to_pandas = _to_pandas

    # as of pyspark 3.2.3, toPandas fails to convert timestamps unless we work around the issue
    # when we upgrade pyspark, we should check whether this workaround is still necessary
    # see https://stackoverflow.com/questions/76389694/transforming-pyspark-to-pandas-dataframe
    if semver.parse(pd.__version__)["major"] >= 2:
        import pyspark.sql.functions as pyspark_functions

        type_conversion_dict = {}
        for field in spark_df.schema.fields:
            if str(field.dataType) == "TimestampType":
                spark_df = spark_df.withColumn(
                    field.name,
                    pyspark_functions.date_format(
                        pyspark_functions.to_timestamp(field.name),
                        "yyyy-MM-dd'T'HH:mm:ss.SSSSSSSSS",
                    ),
                )
                type_conversion_dict[field.name] = "datetime64[ns]"

        df = to_pandas(spark_df)
        if type_conversion_dict:
            df = df.astype(type_conversion_dict)
        return df
    else:
        return to_pandas(spark_df)
