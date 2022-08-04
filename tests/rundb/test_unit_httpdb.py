# test_httpdb.py actually holds integration tests (that should be migrated to tests/integration/sdk_api/httpdb)
# currently we are running it in the integration tests CI step so adding this file for unit tests for the httpdb
import enum
import unittest.mock
import pytest

import mlrun.db.httpdb


class SomeEnumClass(str, enum.Enum):
    value1 = "value1"
    value2 = "value2"


def test_api_call_enum_conversion():
    db = mlrun.db.httpdb.HTTPRunDB("fake-url")
    db.session = unittest.mock.Mock()

    # ensure not exploding when no headers/params
    db.api_call("GET", "some-path")

    db.api_call(
        "GET",
        "some-path",
        headers={"enum-value": SomeEnumClass.value1, "string-value": "value"},
        params={"enum-value": SomeEnumClass.value2, "string-value": "value"},
    )
    for dict_key in ["headers", "params"]:
        for value in db.session.request.call_args_list[1][1][dict_key].values():
            assert type(value) == str


def test_connection_reset_causes_retries():
    db = mlrun.db.httpdb.HTTPRunDB("fake-url")
    db.session = unittest.mock.Mock()
    db.session.request.side_effect = ConnectionResetError

    with unittest.mock.patch("time.sleep") as _:
        with pytest.raises(ConnectionResetError):
            db.api_call("GET", "some-path")

    assert db.session.request.call_count == mlrun.db.httpdb.HTTP_RETRY_AMOUNT
