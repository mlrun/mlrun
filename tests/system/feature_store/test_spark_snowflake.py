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
#
import os
import random
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import snowflake.connector

import mlrun.feature_store as fstore
from mlrun.datastore.sources import SnowflakeSource
from mlrun.datastore.targets import SnowflakeTarget
from mlrun.feature_store.retrieval.spark_merger import spark_df_to_pandas
from tests.system.base import TestMLRunSystem
from tests.system.feature_store.spark_hadoop_test_base import (
    Deployment,
    SparkHadoopTestBase,
)

SNOWFLAKE_ENV_PARAMETERS = [
    "SNOWFLAKE_URL",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_TABLE_NAME",
]


def get_missing_snowflake_spark_parameters():
    snowflake_missing_keys = [
        key for key in SNOWFLAKE_ENV_PARAMETERS if key not in os.environ
    ]
    return snowflake_missing_keys


@TestMLRunSystem.skip_test_if_env_not_configured
class TestSnowFlakeSourceAndTarget(SparkHadoopTestBase):
    @classmethod
    def teardown_class(cls):
        super().teardown_class()
        if cls.snowflake_connector:
            cls.snowflake_connector.close()

    @classmethod
    def custom_setup_class(cls):
        cls.configure_namespace("snowflake")
        cls.env = os.environ
        cls.configure_image_deployment(Deployment.Local)
        snowflake_missing_keys = get_missing_snowflake_spark_parameters()
        if snowflake_missing_keys:
            pytest.skip(
                f"The following snowflake keys are missing: {snowflake_missing_keys}"
            )
        url = cls.env.get("SNOWFLAKE_URL")
        user = cls.env.get("SNOWFLAKE_USER")
        cls.database = cls.env.get("SNOWFLAKE_DATABASE")
        warehouse = cls.env.get("SNOWFLAKE_WAREHOUSE")
        account = url.replace(".snowflakecomputing.com", "")
        password = cls.env["SNOWFLAKE_PASSWORD"]
        cls.snowflake_spark_parameters = dict(
            url=url, user=user, database=cls.database, warehouse=warehouse
        )
        cls.snowflake_connector = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
        )
        cls.schema = cls.env.get("SNOWFLAKE_SCHEMA")
        if cls.deployment_type == Deployment.Remote:
            cls.spark_service = cls.spark_service_name
            cls.run_local = False
        else:
            cls.spark_service = None
            cls.run_local = True

    def setup_method(self, method):
        super().setup_method(method)
        self.cursor = self.snowflake_connector.cursor()
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.source_table = f"source_{self.current_time}"
        self.tables_to_drop = [self.source_table]

    def teardown_method(self, method):
        super().teardown_method(method)
        for table_name in self.tables_to_drop:
            drop_query = (
                f"DROP TABLE IF EXISTS {self.database}.{self.schema}.{table_name}"
            )
            self.cursor.execute(drop_query)
        self.cursor.close()

    def generate_snowflake_source_table(self):
        utc_timezone = timezone.utc

        data_values = [
            (
                i + 1,
                f"Name{i + 1}",
                random.randint(23, 60),
                datetime.utcnow().replace(tzinfo=utc_timezone)
                - timedelta(
                    days=random.randint(2, 1000),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                ),
            )
            for i in range(20)
        ]
        create_table_query = (
            f"CREATE TABLE IF NOT EXISTS {self.database}.{self.schema}.{self.source_table} "
            f"(ID INT,NAME VARCHAR(255),AGE INT, LICENSE_DATE TIMESTAMP_LTZ)"
        )
        self.cursor.execute(create_table_query)
        insert_query = (
            f"INSERT INTO {self.database}.{self.schema}.{self.source_table}"
            f" (ID ,NAME,AGE, LICENSE_DATE) VALUES (%s, %s, %s, %s)"
        )
        self.cursor.executemany(insert_query, data_values)
        return pd.DataFrame(data_values, columns=["ID", "NAME", "AGE", "LICENSE_DATE"])

    def test_snowflake_source_and_target(self):
        number_of_rows = 10
        result_table = f"result_{self.current_time}"
        feature_set = fstore.FeatureSet(
            name="snowflake_feature_set",
            entities=[fstore.Entity("C_CUSTKEY")],
            engine="spark",
        )
        source = SnowflakeSource(
            "snowflake_source_for_ingest",
            query=f"select * from {self.source_table} order by ID limit {number_of_rows}",
            schema=self.schema,
            **self.snowflake_spark_parameters,
        )
        target = SnowflakeTarget(
            "snowflake_target_for_ingest",
            table_name=result_table,
            db_schema=self.schema,
            **self.snowflake_spark_parameters,
        )
        source_df = self.generate_snowflake_source_table()
        self.tables_to_drop.append(result_table)
        feature_set.ingest(
            source,
            targets=[target],
            spark_context=self.spark_service,
            run_config=fstore.RunConfig(local=self.run_local),
        )
        result_data = self.cursor.execute(
            f"select * from {self.database}.{self.schema}.{result_table}"
        ).fetchall()
        column_names = [desc[0] for desc in self.cursor.description]
        result_df = pd.DataFrame(result_data, columns=column_names)
        result_df["LICENSE_DATE"] = result_df["LICENSE_DATE"].dt.tz_convert("UTC")
        expected_df = source_df.sort_values(by="ID").head(number_of_rows)
        pd.testing.assert_frame_equal(expected_df, result_df.sort_values(by="ID"))

    def test_source(self):
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("snowflake_spark")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        number_of_rows = 10

        source = SnowflakeSource(
            "snowflake_source",
            query=f"select * from {self.source_table} order by ID limit {number_of_rows}",
            schema=self.schema,
            time_field="LICENSE_DATE",
            **self.snowflake_spark_parameters,
        )
        source_df = self.generate_snowflake_source_table()
        result_spark_df = source.to_spark_df(session=spark)
        result_df = spark_df_to_pandas(spark_df=result_spark_df)
        sorted_source_df = source_df.sort_values(by="ID").head(number_of_rows)
        pd.testing.assert_frame_equal(
            sorted_source_df,
            result_df,
            check_dtype=False,
        )
