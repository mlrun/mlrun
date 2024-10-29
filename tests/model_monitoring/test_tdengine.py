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

import datetime
from typing import Union

import pytest

import mlrun.common.schemas
from mlrun.model_monitoring.db.tsdb.tdengine.schemas import (
    _MODEL_MONITORING_DATABASE,
    TDEngineSchema,
    _TDEngineColumn,
)

_SUPER_TABLE_TEST = "super_table_test"
_COLUMNS_TEST = {
    "column1": _TDEngineColumn.TIMESTAMP,
    "column2": _TDEngineColumn.FLOAT,
    "column3": _TDEngineColumn.BINARY_40,
}
_TAG_TEST = {"tag1": _TDEngineColumn.INT, "tag2": _TDEngineColumn.BINARY_64}


class TestTDEngineSchema:
    """Tests for the TDEngineSchema class, including the methods to create, insert, delete and query data
    from TDengine."""

    @staticmethod
    @pytest.fixture
    def super_table() -> TDEngineSchema:
        return TDEngineSchema(
            super_table=_SUPER_TABLE_TEST, columns=_COLUMNS_TEST, tags=_TAG_TEST
        )

    @staticmethod
    @pytest.fixture
    def values() -> dict[str, Union[str, int, float, datetime.datetime]]:
        return {
            "column1": datetime.datetime.now(),
            "column2": 0.1,
            "column3": "value3",
            "tag1": 1,
            "tag2": "value2",
        }

    def test_create_super_table(self, super_table: TDEngineSchema):
        assert (
            super_table._create_super_table_query()
            == f"CREATE STABLE if NOT EXISTS {_MODEL_MONITORING_DATABASE}.{super_table.super_table} "
            f"(column1 TIMESTAMP, column2 FLOAT, column3 BINARY(40)) "
            f"TAGS (tag1 INT, tag2 BINARY(64));"
        )

    @pytest.mark.parametrize(
        ("subtable", "remove_tag"), [("subtable_1", False), ("subtable_2", True)]
    )
    def test_create_sub_table(
        self,
        super_table: TDEngineSchema,
        values: dict[str, Union[str, int, float, datetime.datetime]],
        subtable: str,
        remove_tag: bool,
    ):
        assert (
            super_table._create_subtable_sql(subtable=subtable, values=values)
            == f"CREATE TABLE if NOT EXISTS {_MODEL_MONITORING_DATABASE}.{subtable} "
            f"USING {super_table.super_table} TAGS ('{values['tag1']}', '{values['tag2']}');"
        )
        if remove_tag:
            # test with missing tag
            values.pop("tag1")
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                super_table._create_subtable_sql(subtable=subtable, values=values)

    @pytest.mark.parametrize(
        ("subtable", "remove_tag"), [("subtable_1", False), ("subtable_2", True)]
    )
    def test_delete_subtable(
        self,
        super_table: TDEngineSchema,
        values: dict[str, Union[str, int, float, datetime.datetime]],
        subtable: str,
        remove_tag: bool,
    ):
        assert (
            super_table._delete_subtable_query(subtable=subtable, values=values)
            == f"DELETE FROM {_MODEL_MONITORING_DATABASE}.{subtable} "
            f"WHERE tag1 LIKE '{values['tag1']}' AND tag2 LIKE '{values['tag2']}';"
        )

        if remove_tag:
            # test with without one of the tags
            values.pop("tag1")
            assert (
                super_table._delete_subtable_query(subtable=subtable, values=values)
                == f"DELETE FROM {_MODEL_MONITORING_DATABASE}.{subtable} WHERE tag2 LIKE '{values['tag2']}';"
            )

            # test without tags
            values.pop("tag2")
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                super_table._delete_subtable_query(subtable=subtable, values=values)

    def test_drop_subtable(self, super_table: TDEngineSchema):
        assert (
            super_table._drop_subtable_query(subtable="subtable_1")
            == f"DROP TABLE if EXISTS {_MODEL_MONITORING_DATABASE}.subtable_1;"
        )

    @pytest.mark.parametrize(
        ("subtable", "remove_tag"), [("subtable_1", False), ("subtable_2", True)]
    )
    def test_get_subtables(
        self,
        super_table: TDEngineSchema,
        values: dict[str, Union[str, int, float, datetime.datetime]],
        subtable: str,
        remove_tag: bool,
    ):
        assert (
            super_table._get_subtables_query(values=values)
            == f"SELECT tbname FROM {_MODEL_MONITORING_DATABASE}.{super_table.super_table} "
            f"WHERE tag1 LIKE '{values['tag1']}' AND tag2 LIKE '{values['tag2']}';"
        )

        if remove_tag:
            # test with without one of the tags
            values.pop("tag1")
            assert (
                super_table._get_subtables_query(values=values)
                == f"SELECT tbname FROM {_MODEL_MONITORING_DATABASE}.{super_table.super_table} "
                f"WHERE tag2 LIKE '{values['tag2']}';"
            )

            # test without tags
            values.pop("tag2")
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                super_table._get_subtables_query(values=values)

    @pytest.mark.parametrize(
        (
            "subtable",
            "columns_to_filter",
            "filter_query",
            "start",
            "end",
            "timestamp_column",
        ),
        [
            (
                "subtable_1",
                [],
                "",
                mlrun.utils.datetime_now() - datetime.timedelta(hours=1),
                mlrun.utils.datetime_now(),
                "time",
            ),
            (
                "subtable_2",
                ["column2", "column3"],
                "column2 > 0",
                mlrun.utils.datetime_now() - datetime.timedelta(hours=2),
                mlrun.utils.datetime_now() - datetime.timedelta(hours=1),
                "time_column",
            ),
        ],
    )
    def test_get_records_query(
        self,
        super_table: TDEngineSchema,
        subtable: str,
        columns_to_filter: list[str],
        filter_query: str,
        start: str,
        end: str,
        timestamp_column: str,
    ):
        if columns_to_filter:
            columns_to_select = ", ".join(columns_to_filter)
        else:
            columns_to_select = "*"

        if filter_query:
            expected_query = (
                f"SELECT {columns_to_select} FROM {_MODEL_MONITORING_DATABASE}.{subtable} "
                f"WHERE {filter_query} AND {timestamp_column} >= '{start}' "
                f"AND {timestamp_column} <= '{end}';"
            )
        else:
            expected_query = (
                f"SELECT {columns_to_select} FROM {_MODEL_MONITORING_DATABASE}.{subtable} "
                f"WHERE {timestamp_column} >= '{start}' AND {timestamp_column} <= '{end}';"
            )

        assert (
            super_table._get_records_query(
                table=subtable,
                columns_to_filter=columns_to_filter,
                filter_query=filter_query,
                start=start,
                end=end,
                timestamp_column=timestamp_column,
            )
            == expected_query
        )

    @pytest.mark.parametrize(
        (
            "subtable",
            "columns_to_filter",
            "start",
            "end",
            "timestamp_column",
            "interval",
            "limit",
            "agg_funcs",
            "sliding_window_step",
        ),
        [
            (
                "subtable_1",
                ["column2"],
                datetime.datetime.now() - datetime.timedelta(hours=2),
                datetime.datetime.now() - datetime.timedelta(hours=1),
                "time_column",
                "3m",
                2,
                ["count"],
                "1m",
            ),
            (
                "subtable_2",
                ["column2", "column3", "column4", "column5"],
                datetime.datetime.now() - datetime.timedelta(hours=2),
                datetime.datetime.now() - datetime.timedelta(hours=1),
                "time_column_v2",
                "3h",
                50,
                ["avg", "max", "sum"],
                "12m",
            ),
        ],
    )
    def test_get_records_with_interval_query(
        self,
        super_table: TDEngineSchema,
        subtable: str,
        columns_to_filter: list[str],
        start: str,
        end: str,
        timestamp_column: str,
        interval: str,
        limit: int,
        agg_funcs: list,
        sliding_window_step: str,
    ):
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as err:
            # Provide aggregation functions without columns to filter
            super_table._get_records_query(
                table=subtable,
                start=start,
                end=end,
                timestamp_column=timestamp_column,
                interval=interval,
                limit=limit,
                agg_funcs=agg_funcs,
                sliding_window_step=sliding_window_step,
            )
            assert (
                "columns_to_filter must be provided when using aggregate functions"
                in str(err.value)
            )

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as err:
            # Provide interval without aggregation functions
            super_table._get_records_query(
                table=subtable,
                start=start,
                end=end,
                columns_to_filter=columns_to_filter,
                timestamp_column=timestamp_column,
                limit=limit,
                interval=interval,
                sliding_window_step=sliding_window_step,
            )
            assert "`agg_funcs` must be provided when using interval" in str(err.value)

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as err:
            # Provide sliding window without interval
            super_table._get_records_query(
                table=subtable,
                start=start,
                end=end,
                columns_to_filter=columns_to_filter,
                timestamp_column=timestamp_column,
                limit=limit,
                agg_funcs=agg_funcs,
                sliding_window_step=sliding_window_step,
            )
            assert "interval must be provided when using sliding window" in str(
                err.value
            )
        columns_to_select = ", ".join(
            [f"{a}({col})" for a in agg_funcs for col in columns_to_filter]
        )
        expected_query = (
            f""
            f"SELECT _wstart, _wend, {columns_to_select} FROM {_MODEL_MONITORING_DATABASE}.{subtable} "
            f"WHERE {timestamp_column} >= '{start}' AND {timestamp_column} <= '{end}' "
            f"INTERVAL({interval}) SLIDING({sliding_window_step}) LIMIT {limit};"
        )

        assert (
            super_table._get_records_query(
                table=subtable,
                columns_to_filter=columns_to_filter,
                start=start,
                end=end,
                timestamp_column=timestamp_column,
                interval=interval,
                limit=limit,
                agg_funcs=agg_funcs,
                sliding_window_step=sliding_window_step,
            )
            == expected_query
        )
