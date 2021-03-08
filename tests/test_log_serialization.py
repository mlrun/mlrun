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

import datetime
from pprint import pprint

import numpy as np
import pandas as pd

from mlrun import new_task, run_local
from tests.conftest import out_path, tag_test, verify_state


def my_func(context):
    print(f"Run: {context.name} (uid={context.uid})")

    context.log_result("float", 1.5)
    context.log_result("np-float", np.float(1.5))
    context.log_result("np-float32", np.float32(1.5))
    context.log_result("date", datetime.datetime(2018, 1, 1))
    context.log_result("np-date", np.datetime64("2018-01-01"))
    context.log_result("np-nan", np.nan)
    context.log_result("np-list", [1.5, np.nan, np.inf])
    context.log_result("dict", {"x": -1.3, "y": np.float32(1.5), "z": "ab"})
    context.log_result(
        "array", np.array([1, 2, 3.2, np.nan, np.datetime64("2018-01-01")])
    )

    raw_data = {
        "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
        "x": np.array([1, 2, 3.2, np.nan, 5.5]),
        "y": [25, 94, 0.1, 57, datetime.datetime(2018, 1, 1)],
    }
    df = pd.DataFrame(raw_data, columns=["first_name", "last_name", "x", "y"])
    context.log_dataset("df1", df=df, format="csv")

    date_rng = pd.date_range("2018-01-01", periods=4, freq="H")
    df = pd.DataFrame(date_rng, columns=["date"])
    df["data"] = np.random.rand(4)
    df["nan"] = np.nan
    df["datetime"] = pd.to_datetime(df["date"])
    df["text"] = "x"
    df = df.set_index("datetime")
    context.log_dataset("df2", df=df)

    return np.nan


base_spec = new_task(artifact_path=out_path, handler=my_func)


def test_serialization():
    spec = tag_test(base_spec, "test_serialization")
    result = run_local(spec)
    verify_state(result)
    pprint(result.to_dict())
    print(result.to_yaml())
    pprint(result.to_json())
