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
            super_table._create_subtable_query(subtable=subtable, values=values)
            == f"CREATE TABLE if NOT EXISTS {_MODEL_MONITORING_DATABASE}.{subtable} "
            f"USING {super_table.super_table} TAGS ('{values['tag1']}', '{values['tag2']}');"
        )
        if remove_tag:
            # test with missing tag
            values.pop("tag1")
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                super_table._create_subtable_query(subtable=subtable, values=values)

    @pytest.mark.parametrize(
        ("subtable", "remove_value"), [("subtable_1", False), ("subtable_2", True)]
    )
    def test_insert_subtable(
        self,
        super_table: TDEngineSchema,
        values: dict[str, Union[str, int, float, datetime.datetime]],
        subtable: str,
        remove_value: bool,
    ):
        assert (
            super_table._insert_subtable_query(subtable=subtable, values=values)
            == f"INSERT INTO {_MODEL_MONITORING_DATABASE}.{subtable} VALUES ('{values['column1']}', "
            f"'{values['column2']}', '{values['column3']}');"
        )

        if remove_value:
            # test with missing value
            values.pop("column1")
            with pytest.raises(KeyError):
                super_table._insert_subtable_query(subtable=subtable, values=values)

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
            f"WHERE tag1 like '{values['tag1']}' AND tag2 like '{values['tag2']}';"
        )

        if remove_tag:
            # test with without one of the tags
            values.pop("tag1")
            assert (
                super_table._delete_subtable_query(subtable=subtable, values=values)
                == f"DELETE FROM {_MODEL_MONITORING_DATABASE}.{subtable} WHERE tag2 like '{values['tag2']}';"
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
            f"WHERE tag1 like '{values['tag1']}' AND tag2 like '{values['tag2']}';"
        )

        if remove_tag:
            # test with without one of the tags
            values.pop("tag1")
            assert (
                super_table._get_subtables_query(values=values)
                == f"SELECT tbname FROM {_MODEL_MONITORING_DATABASE}.{super_table.super_table} "
                f"WHERE tag2 like '{values['tag2']}';"
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
                datetime.datetime.now().astimezone() - datetime.timedelta(hours=1),
                datetime.datetime.now().astimezone(),
                "time",
            ),
            (
                "subtable_2",
                ["column2", "column3"],
                "column2 > 0",
                datetime.datetime.now().astimezone() - datetime.timedelta(hours=2),
                datetime.datetime.now().astimezone() - datetime.timedelta(hours=1),
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
                f"SELECT {columns_to_select} from {_MODEL_MONITORING_DATABASE}.{subtable} "
                f"where {filter_query} and {timestamp_column} >= '{start}' "
                f"and {timestamp_column} <= '{end}';"
            )
        else:
            expected_query = (
                f"SELECT {columns_to_select} from {_MODEL_MONITORING_DATABASE}.{subtable} "
                f"where {timestamp_column} >= '{start}' and {timestamp_column} <= '{end}';"
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
