# Copyright 2023 Iguazio
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
import os
import random
from datetime import datetime, timedelta

import pandas as pd
import pytest
import snowflake.connector

import mlrun.feature_store as fstore
from mlrun.datastore.sources import SnowflakeSource
from mlrun.datastore.targets import SnowflakeTarget
from tests.system.feature_store.test_spark_engine import TestFeatureStoreSparkEngine

SNOWFLAKE_ENV_PARAMETERS = [
    "SNOWFLAKE_URL",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_TABLE_NAME",
]


class TestSnowFlake(TestFeatureStoreSparkEngine):
    project_name = "fs-system-spark-engine"
    run_local = True  # TODO remove

    @staticmethod
    def get_missing_snowflake_spark_parameters():
        snowflake_missing_keys = [
            key for key in SNOWFLAKE_ENV_PARAMETERS if key not in os.environ
        ]
        return snowflake_missing_keys

    @staticmethod
    def generate_snowflake_table(
        cursor,
        database: str,
        schema: str,
        table_name: str,
    ):
        data_values = [
            (
                i + 1,
                f"Name{i + 1}",
                random.randint(23, 60),
                datetime.now()
                - timedelta(
                    days=random.randint(0, 1000),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                ),
            )
            for i in range(20)
        ]
        create_table_query = (
            f"CREATE TABLE IF NOT EXISTS {database}.{schema}.{table_name} "
            f"(ID INT,NAME VARCHAR(255),AGE INT, LICENSE_DATE TIMESTAMP)"
        )
        cursor.execute(create_table_query)
        insert_query = (
            f"INSERT INTO {database}.{schema}.{table_name}"
            f" (ID ,NAME,AGE, LICENSE_DATE) VALUES (%s, %s, %s, %s)"
        )
        cursor.executemany(insert_query, data_values)
        return pd.DataFrame(data_values, columns=["ID", "NAME", "AGE", "LICENSE_DATE"])

    @staticmethod
    def drop_snowflake_tables(
        cursor,
        database: str,
        schema: str,
        tables: list,
    ):
        for table_name in tables:
            drop_query = f"DROP TABLE IF EXISTS {database}.{schema}.{table_name}"
            cursor.execute(drop_query)

    def test_snowflake_source_and_target(self):
        url = os.environ.get("SNOWFLAKE_URL")
        user = os.environ.get("SNOWFLAKE_USER")
        database = os.environ.get("SNOWFLAKE_DATABASE")
        warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
        password = os.environ["SNOWFLAKE_PASSWORD"]
        schema = os.environ.get("SNOWFLAKE_SCHEMA")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        source_table = f"source_{current_time}"
        result_table = f"result_{current_time}"
        snowflake_spark_parameters = dict(
            url=url, user=user, database=database, warehouse=warehouse
        )
        account = url.replace(".snowflakecomputing.com", "")
        number_of_rows = 10

        snowflake_missing_keys = self.get_missing_snowflake_spark_parameters()
        if snowflake_missing_keys:
            pytest.skip(
                f"The following snowflake keys are missing: {snowflake_missing_keys}"
            )
        self.project.set_secrets({"SNOWFLAKE_PASSWORD": password})
        feature_set = fstore.FeatureSet(
            name="snowflake_feature_set",
            entities=[fstore.Entity("C_CUSTKEY")],
            engine="spark",
        )
        source = SnowflakeSource(
            "snowflake_source",
            query=f"select * from {source_table} order by ID limit {number_of_rows}",
            schema=schema,
            **snowflake_spark_parameters,
        )
        target = SnowflakeTarget(
            "snowflake_target",
            table_name=result_table,
            db_schema=schema,
            **snowflake_spark_parameters,
        )
        ctx = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
        )
        cursor = ctx.cursor()
        try:
            source_df = self.generate_snowflake_table(
                cursor=cursor, database=database, schema=schema, table_name=source_table
            )
            feature_set.ingest(
                source,
                targets=[target],
                spark_context=self.spark_service,
                run_config=fstore.RunConfig(local=self.run_local),
            )
            result_data = cursor.execute(
                f"select * from {database}.{schema}.{result_table}"
            ).fetchall()
            column_names = [desc[0] for desc in cursor.description]
            result_df = pd.DataFrame(result_data, columns=column_names)
            expected_df = source_df.sort_values(by="ID").head(number_of_rows)
            pd.testing.assert_frame_equal(expected_df, result_df.sort_values(by="ID"))
        finally:
            try:
                self.drop_snowflake_tables(
                    cursor=cursor,
                    database=database,
                    schema=schema,
                    tables=[source_table, result_table],
                )
            finally:
                if ctx:
                    ctx.close()
