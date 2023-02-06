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
import unittest.mock

import pandas as pd
import pytest

import mlrun
import mlrun.feature_store as fstore
from mlrun.datastore.targets import DFTarget


def test_columns_with_illegal_characters(rundb_mock):
    df = pd.DataFrame(
        {
            "ticker": ["GOOG", "MSFT"],
            "bid (accepted)": [720.50, 51.95],
            "ask": [720.93, 51.96],
            "with space": [True, False],
        }
    )

    fset = fstore.FeatureSet(
        "myset",
        entities=[fstore.Entity("ticker")],
    )
    fset._run_db = rundb_mock

    fset.reload = unittest.mock.Mock()
    fset.save = unittest.mock.Mock()
    fset.purge_targets = unittest.mock.Mock()

    result_df = fstore.ingest(fset, df, targets=[DFTarget()])
    assert list(result_df.columns) == ["bid_accepted", "ask", "with_space"]


def test_columns_with_illegal_characters_error():
    df = pd.DataFrame(
        {
            "ticker": ["GOOG", "MSFT"],
            "bid (accepted)": [720.50, 51.95],
            "bid_accepted": [720.93, 51.96],
            "with space": [True, False],
        }
    )

    fset = fstore.FeatureSet(
        "myset",
        entities=[fstore.Entity("ticker")],
    )

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        fstore.ingest(fset, df)


def test_set_targets_with_string():
    fset = fstore.FeatureSet(
        "myset",
        entities=[fstore.Entity("ticker")],
    )

    fset.set_targets(["parquet", "nosql"], with_defaults=False)

    targets = fset.spec.targets

    assert len(targets) == 2

    parquet_target = None
    nosql_target = None
    for target in fset.spec.targets:
        if target.name == "parquet":
            parquet_target = target
        elif target.name == "nosql":
            nosql_target = target

    assert parquet_target.name == "parquet"
    assert parquet_target.kind == "parquet"
    assert parquet_target.partitioned

    assert nosql_target.name == "nosql"
    assert nosql_target.kind == "nosql"
    assert not nosql_target.partitioned


def test_return_df(rundb_mock):
    fset = fstore.FeatureSet(
        "myset",
        entities=[fstore.Entity("ticker")],
    )

    df = pd.DataFrame(
        {
            "ticker": ["GOOG", "MSFT"],
            "bid (accepted)": [720.50, 51.95],
            "ask": [720.93, 51.96],
            "with space": [True, False],
        }
    )
    fset._run_db = rundb_mock

    fset.reload = unittest.mock.Mock()
    fset.save = unittest.mock.Mock()
    fset.purge_targets = unittest.mock.Mock()

    result_df = fstore.ingest(fset, df, targets=[DFTarget()], return_df=False)

    assert result_df is None

    result_df = fstore.ingest(fset, df, targets=[DFTarget()])

    assert isinstance(result_df, pd.DataFrame)
