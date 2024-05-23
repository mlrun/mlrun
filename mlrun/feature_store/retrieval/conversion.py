# Copyright 2024 Iguazio
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
import warnings
from collections import Counter

# Copied from https://github.com/apache/spark/blob/v3.2.3/python/pyspark/sql/pandas/conversion.py, with
# np.bool -> bool and np.object -> object fix backported from pyspark v3.3.3.


class PandasConversionMixin:
    """
    Min-in for the conversion from Spark to pandas. Currently, only :class:`DataFrame`
    can use this class.
    """

    def toPandas(self):
        """
        Returns the contents of this :class:`DataFrame` as Pandas ``pandas.DataFrame``.

        This is only available if Pandas is installed and available.

        .. versionadded:: 1.3.0

        Notes
        -----
        This method should only be used if the resulting Pandas's :class:`DataFrame` is
        expected to be small, as all the data is loaded into the driver's memory.

        Usage with spark.sql.execution.arrow.pyspark.enabled=True is experimental.

        Examples
        --------
        >>> df.toPandas()  # doctest: +SKIP
           age   name
        0    2  Alice
        1    5    Bob
        """
        from pyspark.sql.dataframe import DataFrame

        assert isinstance(self, DataFrame)

        from pyspark.sql.pandas.utils import require_minimum_pandas_version

        require_minimum_pandas_version()

        import numpy as np
        import pandas as pd
        from pyspark.sql.types import (
            BooleanType,
            IntegralType,
            MapType,
            TimestampType,
        )

        timezone = self.sql_ctx._conf.sessionLocalTimeZone()

        if self.sql_ctx._conf.arrowPySparkEnabled():
            use_arrow = True
            try:
                from pyspark.sql.pandas.types import to_arrow_schema
                from pyspark.sql.pandas.utils import require_minimum_pyarrow_version

                require_minimum_pyarrow_version()
                to_arrow_schema(self.schema)
            except Exception as e:
                if self.sql_ctx._conf.arrowPySparkFallbackEnabled():
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
                    tmp_column_names = [f"col_{i}" for i in range(len(self.columns))]
                    self_destruct = self.sql_ctx._conf.arrowPySparkSelfDestructEnabled()
                    batches = self.toDF(*tmp_column_names)._collect_as_arrow(
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
                        pdf.columns = self.columns
                        for field in self.schema:
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
                        return pd.DataFrame.from_records([], columns=self.columns)
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
        pdf = pd.DataFrame.from_records(self.collect(), columns=self.columns)
        column_counter = Counter(self.columns)

        dtype = [None] * len(self.schema)
        for field_idx, field in enumerate(self.schema):
            # For duplicate column name, we use `iloc` to access it.
            if column_counter[field.name] > 1:
                pandas_col = pdf.iloc[:, field_idx]
            else:
                pandas_col = pdf[field.name]

            pandas_type = PandasConversionMixin._to_corrected_pandas_type(
                field.dataType
            )
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
            column_name = self.schema[index].name

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
            from pyspark.sql.pandas.types import (
                _check_series_convert_timestamps_local_tz,
            )

            for field in self.schema:
                # TODO: handle nested timestamps, such as ArrayType(TimestampType())?
                if isinstance(field.dataType, TimestampType):
                    pdf[field.name] = _check_series_convert_timestamps_local_tz(
                        pdf[field.name], timezone
                    )
            return pdf

    @staticmethod
    def _to_corrected_pandas_type(dt):
        """
        When converting Spark SQL records to Pandas :class:`DataFrame`, the inferred data type
        may be wrong. This method gets the corrected data type for Pandas if that type may be
        inferred incorrectly.
        """
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
            return bool
        elif type(dt) == TimestampType:
            return np.datetime64
        else:
            return None
