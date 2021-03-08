import os
import random
import string

import pandas as pd
import pytest
from data_sample import stocks
from storey import ReadCSV, ReduceToDataFrame, build_flow

import mlrun.feature_store as fs
from mlrun.config import config as mlconf
from mlrun.data_types.data_types import ValueType
from mlrun.datastore.targets import CSVTarget
from mlrun.feature_store import Entity
from tests.conftest import results, tests_root_directory

local_dir = f"{tests_root_directory}/feature-store/"
results_dir = f"{results}/feature-store/"


def init_store():
    mlconf.dbpath = os.environ["TEST_DBPATH"]


def has_db():
    return "TEST_DBPATH" in os.environ


def _generate_random_name():
    random_name = "".join([random.choice(string.ascii_letters) for i in range(10)])
    return random_name


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_read_csv():
    csv_path = results_dir + _generate_random_name() + ".csv"

    init_store()

    targets = [CSVTarget("mycsv", path=csv_path)]
    stocks_set = fs.FeatureSet("tests", entities=[Entity("ticker", ValueType.STRING)])
    fs.ingest(
        stocks_set, stocks, infer_options=fs.InferOptions.default(), targets=targets
    )

    # reading csv file
    controller = build_flow([ReadCSV(csv_path), ReduceToDataFrame()]).run()
    termination_result = controller.await_termination()

    expected = pd.DataFrame(
        {
            0: ["ticker", "MSFT", "GOOG", "AAPL"],
            1: ["name", "Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
            2: ["exchange", "NASDAQ", "NASDAQ", "NASDAQ"],
        }
    )

    assert termination_result.equals(expected), f"{termination_result}\n!=\n{expected}"

    os.remove(csv_path)
