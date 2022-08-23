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


def test_set_targets_with_string():
    fset = fs.FeatureSet(
        "myset",
        entities=[fs.Entity("ticker")],
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
