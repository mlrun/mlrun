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
# test_httpdb.py actually holds integration tests (that should be migrated to tests/integration/sdk_api/httpdb)
# currently we are running it in the integration tests CI step so adding this file for unit tests for the httpdb
import enum
import unittest.mock

import pandas as pd
import pytest
import requests

import mlrun.config
import mlrun.data_types.data_types as dt
import mlrun.db.httpdb
import mlrun.feature_store as fstore


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


@pytest.mark.parametrize(
    "feature_config,exception_type,exception_message,call_amount",
    [
        # feature enabled
        ("enabled", Exception, "some-error", 1),
        ("enabled", ConnectionError, "some-error", 1),
        ("enabled", ConnectionResetError, "some-error", 1),
        (
            "enabled",
            ConnectionError,
            "Connection aborted",
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "enabled",
            ConnectionResetError,
            "Connection reset by peer",
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        (
            "enabled",
            ConnectionRefusedError,
            "Connection refused",
            # one try + the max retries
            1 + mlrun.config.config.http_retry_defaults.max_retries,
        ),
        # feature disabled
        ("disabled", Exception, "some-error", 1),
        ("disabled", ConnectionError, "some-error", 1),
        ("disabled", ConnectionResetError, "some-error", 1),
        ("disabled", ConnectionError, "Connection aborted", 1),
        (
            "disabled",
            ConnectionResetError,
            "Connection reset by peer",
            1,
        ),
        (
            "disabled",
            ConnectionRefusedError,
            "Connection refused",
            1,
        ),
    ],
)
def test_connection_reset_causes_retries(
    feature_config, exception_type, exception_message, call_amount
):
    mlrun.config.config.httpdb.retry_api_call_on_exception = feature_config
    db = mlrun.db.httpdb.HTTPRunDB("fake-url")
    original_request = requests.Session.request
    requests.Session.request = unittest.mock.Mock()
    requests.Session.request.side_effect = exception_type(exception_message)

    # patch sleep to make test faster
    with unittest.mock.patch("time.sleep"):
        with pytest.raises(exception_type):
            db.api_call("GET", "some-path")

    assert requests.Session.request.call_count == call_amount
    requests.Session.request = original_request


def test_get_projects_dataschemas():
    # prepare data
    #   create test project
    db = mlrun.get_run_db()
    mlrun.get_or_create_project("test-dataschemas", context="./", user_project=False)

    #   create feature set 01
    feature_set = fstore.FeatureSet(
        "FeatureSet01-DataSchema",
        entities=[
            mlrun.feature_store.Entity(
                "ds01fn0", description="fn0 featureset01 description"
            ),
            fstore.Entity("ds01fn1", value_type=dt.ValueType.STRING),
        ],
    )
    pt = mlrun.datastore.ParquetTarget(name="tst", path="./tmp/tst/")
    feature_set.set_targets(targets=[pt], with_defaults=False)
    df = pd.DataFrame(
        [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)],
        columns=["ds01fn0", "ds01fn1", "ds01fn2", "ds01fn3"],
    )
    fstore.ingest(feature_set, df)
    feature_set.save()

    #   create feature set 02
    feature_set = fstore.FeatureSet(
        "FeatureSet02-DataSchema",
        entities=[
            mlrun.feature_store.Entity(
                "ds02fn0", description="fn0 featureset02 description"
            ),
            fstore.Entity("ds02fn1", value_type=dt.ValueType.FLOAT),
        ],
    )
    pt = mlrun.datastore.ParquetTarget(name="tst", path="./tmp/tst/")
    feature_set.set_targets(targets=[pt], with_defaults=False)
    df = pd.DataFrame(
        [(1.1, 2.2), (3.3, 4.4), (5.5, 6.6)], columns=["ds02fn0", "ds02fn1"]
    )
    fstore.ingest(feature_set, df)
    feature_set.save()

    # tests
    dataschemas = db.get_projects_dataschemas()

    assert dataschemas is not None, "Empty output"
    assert len(dataschemas) >= 1, "Missing project"

    dataschema_tests = [
        {
            "test": "FeatureSet01-DataSchema",
            "result": True,
            "err": "Missing featureset",
        },
        {
            "test": "FeatureSet02-DataSchema",
            "result": True,
            "err": "Missing featureset",
        },
        {
            "test": "FeatureSet03-DataSchema",
            "result": False,
            "err": "Unwanted featureset",
        },
        {"test": "ds01fn0", "result": True, "err": "Missing entity"},
        {"test": "ds01fn1", "result": True, "err": "Missing entity"},
        {"test": "ds02fn0", "result": True, "err": "Missing entity"},
        {"test": "ds02fn1", "result": True, "err": "Missing entity"},
        {"test": "ds01fn2", "result": True, "err": "Missing feature"},
        {"test": "ds01fn3", "result": True, "err": "Missing feature"},
        {"test": "ds01fn4", "result": False, "err": "Missing feature"},
        {
            "test": "fn0 featureset01 description",
            "result": True,
            "err": "Missing description",
        },
        {
            "test": "fn0 featureset02 description",
            "result": True,
            "err": "Missing description",
        },
        {
            "test": "fn0 featureset03 description",
            "result": False,
            "err": "Missing description",
        },
    ]

    dataschemas_text = str(dataschemas)
    for test_item in dataschema_tests:
        if test_item["result"]:
            assert dataschemas_text.find(test_item["test"]) != -1, "{0}: '{1}'".format(
                test_item["err"], test_item["test"]
            )
        else:
            assert dataschemas_text.find(test_item["test"]) == -1, "{0}: '{1}'".format(
                test_item["err"], test_item["test"]
            )
