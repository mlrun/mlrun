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

import mlrun.errors
import mlrun.feature_store as fstore
from mlrun.datastore.sources import SnowflakeSource
from mlrun.datastore.targets import ParquetTarget, SnowflakeTarget
from tests.system.base import TestMLRunSystem
from tests.system.feature_store.spark_hadoop_test_base import (
    Deployment,
    SparkHadoopTestBase,
)
from tests.system.feature_store.utils import (
    get_missing_snowflake_spark_parameters,
    get_snowflake_spark_parameters,
)


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
        cls.configure_image_deployment(Deployment.Remote)
        snowflake_missing_keys = get_missing_snowflake_spark_parameters()
        if snowflake_missing_keys:
            pytest.skip(
                f"The following snowflake keys are missing: {snowflake_missing_keys}"
            )
        cls.snowflake_spark_parameters = get_snowflake_spark_parameters()
        cls.database = cls.snowflake_spark_parameters["database"]
        account = cls.snowflake_spark_parameters["url"].replace(
            ".snowflakecomputing.com", ""
        )
        cls.snowflake_connector = snowflake.connector.connect(
            account=account,
            user=cls.snowflake_spark_parameters["user"],
            password=cls.env["SNOWFLAKE_PASSWORD"],
            warehouse=cls.snowflake_spark_parameters["warehouse"],
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
        if self.deployment_type == Deployment.Remote:
            self.project.set_secrets(
                {"SNOWFLAKE_PASSWORD": os.environ["SNOWFLAKE_PASSWORD"]}
            )

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
                (i + 1) * 10,
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

    @pytest.mark.parametrize("passthrough", [True, False])
    def test_snowflake_source_and_target(self, passthrough):
        number_of_rows = 10
        result_table = f"result_{self.current_time}"
        feature_set = fstore.FeatureSet(
            name="snowflake_feature_set",
            entities=[fstore.Entity("ID")],
            engine="spark",
            passthrough=passthrough,
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
            run_config=fstore.RunConfig(
                local=self.run_local,
            ),
        )
        expected_df = source_df.sort_values(by="ID").head(number_of_rows)
        if not passthrough:
            result_data = self.cursor.execute(
                f"select * from {self.database}.{self.schema}.{result_table}"
            ).fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            result_df = pd.DataFrame(result_data, columns=column_names)
            result_df["LICENSE_DATE"] = result_df["LICENSE_DATE"].dt.tz_convert("UTC")

            pd.testing.assert_frame_equal(expected_df, result_df.sort_values(by="ID"))
        vector = fstore.FeatureVector(
            "feature_vector_snowflake", ["snowflake_feature_set.*"]
        )
        run_config = fstore.RunConfig(
            local=self.run_local, kind=None if self.run_local else "remote-spark"
        )
        result = vector.get_offline_features(
            engine="spark",
            with_indexes=True,
            spark_service=self.spark_service,
            run_config=run_config,
            target=None if self.run_local else ParquetTarget(),
        ).to_dataframe()
        result = result.reset_index(drop=False)
        pd.testing.assert_frame_equal(
            expected_df, result.sort_values(by="ID"), check_dtype=False
        )

    def test_feature_set_wrong_letters_case(self):
        from mlrun import feature_store as fs
        from mlrun.feature_store import Entity

        self.generate_snowflake_source_table()
        result_table = f"result_{self.current_time}"
        self.tables_to_drop.append(result_table)

        source = SnowflakeSource(
            "snowflake_target_for_ingest",
            query=f"select * from {self.source_table} order by ID limit 10",
            db_schema=self.schema,
            **self.snowflake_spark_parameters,
        )

        target = SnowflakeTarget(
            "snowflake_target_for_ingest",
            table_name=result_table,
            db_schema=self.schema,
            **self.snowflake_spark_parameters,
        )
        timestamp_key_feature_store = fs.FeatureSet(
            "timestamp_key_feature_store",
            timestamp_key="license_date",
            engine="spark",
            passthrough=False,
            relations=None,
        )
        entity_feature_store = fs.FeatureSet(
            "entity_feature_store",
            entities=[Entity("id")],
            engine="spark",
            passthrough=False,
            relations=None,
        )
        run_config = fstore.RunConfig(local=self.run_local)
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError,
                           match="Snowflake supports timestamp_key as upper case only"):
            timestamp_key_feature_store.ingest(source, [target], run_config=run_config, spark_context=self.spark_service)
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError,
                           match="Snowflake supports entity as upper case only"):
            entity_feature_store.ingest(source, [target], run_config=run_config, spark_context=self.spark_service)
