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
import mlrun.feature_store as fs
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

    fset = fs.FeatureSet(
        "myset",
        entities=[fs.Entity("ticker")],
    )
    fset._run_db = rundb_mock

    fset.reload = unittest.mock.Mock()
    fset.save = unittest.mock.Mock()
    fset.purge_targets = unittest.mock.Mock()

    result_df = fs.ingest(fset, df, targets=[DFTarget()])
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

    fset = fs.FeatureSet(
        "myset",
        entities=[fs.Entity("ticker")],
    )

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        fs.ingest(fset, df)
