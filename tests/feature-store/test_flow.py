import os
from http import HTTPStatus

from storey.dtypes import V3ioError

import mlrun
import pytest
import pandas as pd
import asyncio
import random
import string
import aiohttp
import os
from tests.conftest import results, tests_root_directory

from storey import MapClass, ReadCSV, build_flow, ReduceToDataFrame

from mlrun.datastore.targets import CSVTarget
from mlrun.utils import logger
import mlrun.feature_store as fs
from mlrun.config import config as mlconf
from mlrun.feature_store import FeatureSet, Entity, run_ingestion_job
from mlrun.data_types.data_types import ValueType

local_dir = f"{tests_root_directory}/feature-store/"
results_dir = f"{results}/feature-store/"


def init_store():
    mlconf.dbpath = os.environ["TEST_DBPATH"]


def has_db():
    return "TEST_DBPATH" in os.environ


def _generate_random_name():
    random_name = ''.join([random.choice(string.ascii_letters) for i in range(10)])
    return random_name


@pytest.mark.skipif(not has_db(), reason="no db access")
def test_read_csv():

    csv_path = results_dir + _generate_random_name() + ".csv"

    init_store()

    stocks = pd.DataFrame(
        {
            "ticker": ["MSFT", "GOOG", "AAPL"],
            "name": ["Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
            "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
        }
    )

    targets = [CSVTarget("mycsv", path=csv_path)]
    stocks_set = fs.FeatureSet("tests", entities=[Entity("ticker", ValueType.STRING)])
    fs.ingest(stocks_set, stocks, infer_options=fs.InferOptions.default(), targets=targets)

    # reading csv file
    controller = build_flow([
        ReadCSV(csv_path),
        ReduceToDataFrame()
    ]).run()
    termination_result = controller.await_termination()

    expected = pd.DataFrame({0: ['ticker', 'MSFT', 'GOOG', 'AAPL'], 1: ['name', "Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
                             2: ['exchange', "NASDAQ", "NASDAQ", "NASDAQ"]})

    assert termination_result.equals(expected), f"{termination_result}\n!=\n{expected}"

    os.remove(csv_path)

