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


import os
from typing import Union

import pandas as pd

SNOWFLAKE_ENV_PARAMETERS = [
    "SNOWFLAKE_URL",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_TABLE_NAME",
]


def sort_df(df: pd.DataFrame, sort_columns: Union[str, list[str]]):
    return (
        df.reindex(sorted(df.columns), axis=1)
        .sort_values(by=sort_columns)
        .reset_index(drop=True)
    )


def get_missing_snowflake_spark_parameters():
    snowflake_missing_keys = [
        key for key in SNOWFLAKE_ENV_PARAMETERS if key not in os.environ
    ]
    return snowflake_missing_keys


def get_snowflake_spark_parameters():
    url = os.environ.get("SNOWFLAKE_URL")
    user = os.environ.get("SNOWFLAKE_USER")
    database = os.environ.get("SNOWFLAKE_DATABASE")
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
    return dict(url=url, user=user, database=database, warehouse=warehouse)
