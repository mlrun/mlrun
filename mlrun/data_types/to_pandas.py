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

from collections import Counter
from typing import List, Optional, Type
from warnings import catch_warnings, simplefilter, warn

from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DataType,
    DayTimeIntervalType,
    DoubleType,
    FloatType,
    IntegerType,
    IntegralType,
    LongType,
    MapType,
    ShortType,
    TimestampNTZType,
    TimestampType,
    Union,
)


def toPandas(spark_df):
    """
    Modified version of spark DataFrame.toPandas() –
    https://github.com/apache/spark/blob/v3.2.3/python/pyspark/sql/pandas/conversion.py#L35

    The original code (which is only replaced in pyspark 3.5.0) fails with Pandas 2 installed, with the following error:
    Casting to unit-less dtype 'datetime64' is not supported. Pass e.g. 'datetime64[ns]' instead.

    This modification adds the missing unit to the dtype.
    """

    from pyspark.sql.dataframe import DataFrame

    assert isinstance(spark_df, DataFrame)

    from pyspark.sql.pandas.utils import require_minimum_pandas_version

    require_minimum_pandas_version()

    import numpy as np
    import pandas as pd
    from pandas.core.dtypes.common import is_timedelta64_dtype

    jconf = spark_df.sparkSession._jconf
    timezone = jconf.sessionLocalTimeZone()

    if jconf.arrowPySparkEnabled():
        use_arrow = True
        try:
            from pyspark.sql.pandas.types import to_arrow_schema
            from pyspark.sql.pandas.utils import require_minimum_pyarrow_version

            require_minimum_pyarrow_version()
            to_arrow_schema(spark_df.schema)
        except Exception as e:

            if jconf.arrowPySparkFallbackEnabled():
                msg = (
                    "toPandas attempted Arrow optimization because "
                    "'spark.sql.execution.arrow.pyspark.enabled' is set to true; however, "
                    "failed by the reason below:\n  %s\n"
                    "Attempting non-optimization as "
                    "'spark.sql.execution.arrow.pyspark.fallback.enabled' is set to "
                    "true." % str(e)
                )
                warn(msg)
                use_arrow = False
            else:
                msg = (
                    "toPandas attempted Arrow optimization because "
                    "'spark.sql.execution.arrow.pyspark.enabled' is set to true, but has "
                    "reached the error below and will not continue because automatic fallback "
                    "with 'spark.sql.execution.arrow.pyspark.fallback.enabled' has been set to "
                    "false.\n  %s" % str(e)
                )
                warn(msg)
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
                tmp_column_names = [
                    "col_{}".format(i) for i in range(len(spark_df.columns))
                ]
                self_destruct = jconf.arrowPySparkSelfDestructEnabled()
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
                    corrected_panda_types = {}
                    for index, field in enumerate(spark_df.schema):
                        pandas_type = _to_corrected_pandas_type(field.dataType)
                        corrected_panda_types[tmp_column_names[index]] = (
                            np.object0 if pandas_type is None else pandas_type
                        )

                    pdf = pd.DataFrame(columns=tmp_column_names).astype(
                        dtype=corrected_panda_types
                    )
                    pdf.columns = spark_df.columns
                    return pdf
            except Exception as e:
                # We might have to allow fallback here as well but multiple Spark jobs can
                # be executed. So, simply fail in this case for now.
                msg = (
                    "toPandas attempted Arrow optimization because "
                    "'spark.sql.execution.arrow.pyspark.enabled' is set to true, but has "
                    "reached the error below and can not continue. Note that "
                    "'spark.sql.execution.arrow.pyspark.fallback.enabled' does not have an "
                    "effect on failures in the middle of "
                    "computation.\n  %s" % str(e)
                )
                warn(msg)
                raise

    # Below is toPandas without Arrow optimization.
    pdf = pd.DataFrame.from_records(spark_df.collect(), columns=spark_df.columns)
    column_counter = Counter(spark_df.columns)

    corrected_dtypes: List[Optional[Type]] = [None] * len(spark_df.schema)
    for index, field in enumerate(spark_df.schema):
        # We use `iloc` to access columns with duplicate column names.
        if column_counter[field.name] > 1:
            pandas_col = pdf.iloc[:, index]
        else:
            pandas_col = pdf[field.name]

        pandas_type = _to_corrected_pandas_type(field.dataType)
        # SPARK-21766: if an integer field is nullable and has null values, it can be
        # inferred by pandas as a float column. If we convert the column with NaN back
        # to integer type e.g., np.int16, we will hit an exception. So we use the
        # pandas-inferred float type, rather than the corrected type from the schema
        # in this case.
        if pandas_type is not None and not (
            isinstance(field.dataType, IntegralType)
            and field.nullable
            and pandas_col.isnull().any()
        ):
            corrected_dtypes[index] = pandas_type
        # Ensure we fall back to nullable numpy types.
        if isinstance(field.dataType, IntegralType) and pandas_col.isnull().any():
            corrected_dtypes[index] = np.float64
        if isinstance(field.dataType, BooleanType) and pandas_col.isnull().any():
            corrected_dtypes[index] = np.object  # type: ignore[attr-defined]

    df = pd.DataFrame()
    for index, t in enumerate(corrected_dtypes):
        column_name = spark_df.schema[index].name

        # We use `iloc` to access columns with duplicate column names.
        if column_counter[column_name] > 1:
            series = pdf.iloc[:, index]
        else:
            series = pdf[column_name]

        # No need to cast for non-empty series for timedelta. The type is already correct.
        should_check_timedelta = is_timedelta64_dtype(t) and len(pdf) == 0

        if (t is not None and not is_timedelta64_dtype(t)) or should_check_timedelta:
            series = series.astype(t, copy=False)

        with catch_warnings():
            from pandas.errors import PerformanceWarning

            simplefilter(action="ignore", category=PerformanceWarning)
            # `insert` API makes copy of data,
            # we only do it for Series of duplicate column names.
            # `pdf.iloc[:, index] = pdf.iloc[:, index]...` doesn't always work
            # because `iloc` could return a view or a copy depending by context.
            if column_counter[column_name] > 1:
                df.insert(index, column_name, series, allow_duplicates=True)
            else:
                df[column_name] = series

    if timezone is None:
        return df
    else:
        from pyspark.sql.pandas.types import _check_series_convert_timestamps_local_tz

        for field in spark_df.schema:
            # TODO: handle nested timestamps, such as ArrayType(TimestampType())?
            if isinstance(field.dataType, TimestampType):
                df[field.name] = _check_series_convert_timestamps_local_tz(
                    df[field.name], timezone
                )
        return df


def _to_corrected_pandas_type(dt: DataType) -> Union[None, Type, str]:
    import numpy as np

    if type(dt) == ByteType:
        return np.int8
    elif type(dt) == ShortType:
        return np.int16
    elif type(dt) == IntegerType:
        return np.int32
    elif type(dt) == LongType:
        return np.int64
    elif type(dt) == FloatType:
        return np.float32
    elif type(dt) == DoubleType:
        return np.float64
    elif type(dt) == BooleanType:
        return np.bool  # type: ignore[attr-defined]
    elif type(dt) == TimestampType:
        return "datetime64[ns]"
    elif type(dt) == TimestampNTZType:
        return "datetime64[ns]"
    elif type(dt) == DayTimeIntervalType:
        return np.timedelta64
    else:
        return None
