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

import pandas as pd
import pytest
import sqlalchemy as db

import mlrun.feature_store as fs
from mlrun.datastore.sources import SqlDBSource
from mlrun.datastore.targets import SqlDBTarget
from mlrun.feature_store.steps import OneHotEncoder
from tests.system.base import TestMLRunSystem

CREDENTIALS_ENV = "MLRUN_SYSTEM_TESTS_MONGODB_CONNECTION_STRING"


def _are_mongodb_connection_string_not_set() -> bool:
    return False


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.skipif(
    _are_mongodb_connection_string_not_set(),
    reason=f"Environment variable {CREDENTIALS_ENV} is not defined",
)
@pytest.mark.enterprise
class TestFeatureStoreMongoDB(TestMLRunSystem):
    project_name = "fs-system-test-sqldb"

    @classmethod
    def _init_env_from_file(cls):
        env = cls._get_env_from_file()
        cls.db = env["SQL_DB_PATH_STRING"]
        cls.source_collection = "source_collection"
        cls.target_collection = "target_collection"

    def custom_setup(self):
        self._init_env_from_file()
        self.prepare_data()

    def get_data(self, data_name: str):
        if data_name == "stocks":
            return self.stocks
        elif data_name == "quotes":
            return self.quotes
        elif data_name == "trades":
            return self.trades
        else:
            return None

    @staticmethod
    def get_schema(data_name: str):
        if data_name == "stocks":
            return {"ticker": str, "name": str, "exchange": str}
        elif data_name == "quotes":
            return {
                "time": datetime.datetime,
                "ticker": str,
                "bid": float,
                "ask": float,
                "ind": int,
            }
        elif data_name == "trades":
            return {
                "time": datetime.datetime,
                "ticker": str,
                "price": float,
                "quantity": int,
                "ind": int,
            }
        else:
            return None

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        yield
        # drop all the collection on self.db
        engine = db.create_engine(self.db)
        sql_connection = engine.connect()
        metadata = db.MetaData()
        metadata.reflect(bind=engine)
        # and drop them, if they exist
        metadata.drop_all(bind=engine, checkfirst=True)
        engine.dispose()
        sql_connection.close()

    @pytest.mark.parametrize(
        "source_name, key", [("stocks", "ticker"), ("trades", "ind"), ("quotes", "ind")]
    )
    def test_sql_source_basic(self, source_name: str, key: str):
        engine = db.create_engine(self.db, echo=True)
        sqlite_connection = engine.connect()
        origin_df = self.get_data(source_name)
        origin_df.to_sql(source_name, sqlite_connection, if_exists="replace")
        sqlite_connection.close()
        source = SqlDBSource(
            table_name=source_name, db_path=self.db, key_field=key
        )

        feature_set = fs.FeatureSet(f"fs-{source_name}", entities=[fs.Entity(key)])
        df = fs.ingest(feature_set, source=source)
        df.reset_index(drop=True, inplace=True)
        origin_df.drop(columns=[key], inplace=True)
        origin_df.reset_index(drop=True, inplace=True)
        assert df.equals(origin_df)

    @pytest.mark.parametrize(
        "source_name, key, encoder_col",
        [
            ("stocks", "ticker", "exchange"),
            ("trades", "ind", "ticker"),
            ("quotes", "ind", "ticker"),
        ],
    )
    def test_sql_source_with_step(self, source_name: str, key: str, encoder_col: str):
        engine = db.create_engine(self.db, echo=True)
        sqlite_connection = engine.connect()
        origin_df = self.get_data(source_name)
        origin_df.to_sql(source_name, sqlite_connection, if_exists="replace")
        sqlite_connection.close()

        # test source
        source = SqlDBSource(
            table_name=source_name, db_path=self.db, key_field=key
        )
        feature_set = fs.FeatureSet(f"fs-{source_name}", entities=[fs.Entity(key)])
        one_hot_encoder_mapping = {
            encoder_col: list(origin_df[encoder_col].unique()),
        }
        feature_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
        df = fs.ingest(feature_set, source=source)

        # reference source
        feature_set_ref = fs.FeatureSet(
            f"fs-{source_name}-ref", entities=[fs.Entity(key)]
        )
        feature_set_ref.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
        df_ref = fs.ingest(feature_set_ref, origin_df)

        assert df.equals(df_ref)

    @pytest.mark.parametrize(
        "source_name, key, aggr_col",
        [("quotes", "ind", "ask"), ("trades", "ind", "price")],
    )
    def test_sql_source_with_aggregation(
        self, source_name: str, key: str, aggr_col: str
    ):
        engine = db.create_engine(self.db, echo=True)
        sqlite_connection = engine.connect()
        origin_df = self.get_data(source_name)
        origin_df.to_sql(source_name, sqlite_connection, if_exists="replace")
        sqlite_connection.close()

        # test source
        source = SqlDBSource(
            table_name=source_name, db_path=self.db, key_field=key
        )
        feature_set = fs.FeatureSet(f"fs-{source_name}", entities=[fs.Entity(key)])
        feature_set.add_aggregation(
            aggr_col, ["sum", "max"], "1h", "10m", name=f"{aggr_col}1"
        )
        df = fs.ingest(feature_set, source=source)

        # reference source
        feature_set_ref = fs.FeatureSet(
            f"fs-{source_name}-ref", entities=[fs.Entity(key)]
        )
        feature_set_ref.add_aggregation(
            aggr_col, ["sum", "max"], "1h", "10m", name=f"{aggr_col}1"
        )
        df_ref = fs.ingest(feature_set_ref, origin_df)

        assert df.equals(df_ref)

    @pytest.mark.parametrize(
        "target_name, key", [("stocks", "ticker"), ("trades", "ind"), ("quotes", "ind")]
    )
    def test_sql_target_basic(self, target_name: str, key: str):
        origin_df = self.get_data(target_name)
        schema = self.get_schema(target_name)

        target = SqlDBTarget(
            table_name=target_name,
            db_path=self.db,
            create_table=True,
            schema=schema,
            primary_key_column=key,
        )
        feature_set = fs.FeatureSet(f"fs-{target_name}-tr", entities=[fs.Entity(key)])
        fs.ingest(feature_set, source=origin_df, targets=[target])
        df = target.as_df()

        origin_df.set_index(key, inplace=True)
        columns = [*schema.keys()]
        columns.remove(key)
        df.sort_index(inplace=True), origin_df.sort_index(inplace=True)

        assert df[columns].equals(origin_df[columns])

    @pytest.mark.parametrize(
        "target_name, key", [("stocks", "ticker"), ("trades", "ind"), ("quotes", "ind")]
    )
    def test_sql_target_without_create(self, target_name: str, key: str):
        origin_df = self.get_data(target_name)
        schema = self.get_schema(target_name)
        engine = db.create_engine(self.db, echo=True)
        sqlite_connection = engine.connect()
        metadata = db.MetaData()
        self._create(schema, target_name, metadata, engine, key)
        sqlite_connection.close()

        target = SqlDBTarget(
            table_name=target_name,
            db_path=self.db,
            create_table=False,
            primary_key_column=key,
        )
        feature_set = fs.FeatureSet(f"fs-{target_name}-tr", entities=[fs.Entity(key)])
        fs.ingest(feature_set, source=origin_df, targets=[target])
        df = target.as_df()

        origin_df.set_index(key, inplace=True)
        columns = [*schema.keys()]
        columns.remove(key)
        df.sort_index(inplace=True), origin_df.sort_index(inplace=True)

        assert df[columns].equals(origin_df[columns])

    @pytest.mark.parametrize("target_name, key", [("trades", "ind"), ("quotes", "ind")])
    def test_sql_get_online_feature_basic(self, target_name: str, key: str):
        origin_df = self.get_data(target_name)
        schema = self.get_schema(target_name)

        target = SqlDBTarget(
            table_name=target_name,
            db_path=self.db,
            create_table=True,
            schema=schema,
            primary_key_column=key,
        )
        feature_set = fs.FeatureSet(f"fs-{target_name}-tr", entities=[fs.Entity(key)])
        feature_set_ref = fs.FeatureSet(
            f"fs-{target_name}-ref", entities=[fs.Entity(key)]
        )
        fs.ingest(feature_set, source=origin_df, targets=[target])
        fs.ingest(feature_set_ref, source=origin_df)
        columns = [*schema.keys()]
        columns.remove(key)

        # reference
        features_ref = [
            f"fs-{target_name}-ref.{columns[-1]}",
            f"fs-{target_name}-ref.{columns[-2]}",
        ]
        vector = fs.FeatureVector(
            f"{target_name}-vec", features_ref, description="my test vector"
        )
        service_ref = fs.get_online_feature_service(vector)
        ref_output = service_ref.get([{key: 1}], as_list=True)

        # test
        features = [
            f"fs-{target_name}-tr.{columns[-1]}",
            f"fs-{target_name}-tr.{columns[-2]}",
        ]
        vector = fs.FeatureVector(
            f"{target_name}-vec", features, description="my test vector"
        )
        service = fs.get_online_feature_service(vector)
        output = service.get([{key: 1}], as_list=True)

        assert ref_output == output

    @pytest.mark.parametrize(
        "name, key", [("stocks", "ticker"), ("trades", "ind"), ("quotes", "ind")]
    )
    def test_sql_source_and_target_basic(self, name: str, key: str):
        origin_df = self.get_data(name)
        schema = self.get_schema(name)

        engine = db.create_engine(self.db, echo=True)
        sqlite_connection = engine.connect()
        origin_df.to_sql(f"{name}-source", sqlite_connection, if_exists="replace")
        sqlite_connection.close()

        source = SqlDBSource(
            table_name=f"{name}-source", db_path=self.db, key_field=key
        )

        target = SqlDBTarget(
            table_name=f"{name}-tatget",
            db_path=self.db,
            create_table=True,
            schema=schema,
            primary_key_column=key,
        )

        targets = [target]
        feature_set = fs.FeatureSet(
            "sample_training_posts",
            entities=[fs.Entity(key)],
            description="feature set",
        )

        ingest_df = fs.ingest(
            feature_set,
            source=source,
            targets=targets,
        )

        ingest_df.reset_index(drop=True, inplace=True)
        origin_df.drop(columns=[key], inplace=True)
        origin_df.reset_index(drop=True, inplace=True)
        assert ingest_df.equals(origin_df)

    def prepare_data(self):
        import pandas as pd

        self.quotes = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2016-05-25 13:30:00.023"),
                    pd.Timestamp("2016-05-25 13:30:00.023"),
                    pd.Timestamp("2016-05-25 13:30:00.030"),
                    pd.Timestamp("2016-05-25 13:30:00.041"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                    pd.Timestamp("2016-05-25 13:30:00.049"),
                    pd.Timestamp("2016-05-25 13:30:00.072"),
                    pd.Timestamp("2016-05-25 13:30:00.075"),
                ],
                "ticker": [
                    "GOOG",
                    "MSFT",
                    "MSFT",
                    "MSFT",
                    "GOOG",
                    "AAPL",
                    "GOOG",
                    "MSFT",
                ],
                "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
                "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
                "ind": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        self.trades = pd.DataFrame(
            {
                "time": [
                    pd.Timestamp("2016-05-25 13:30:00.023"),
                    pd.Timestamp("2016-05-25 13:30:00.038"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                    pd.Timestamp("2016-05-25 13:30:00.048"),
                ],
                "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
                "price": [51.95, 51.95, 720.77, 720.92, 98.0],
                "quantity": [75, 155, 100, 100, 100],
                "ind": [1, 2, 3, 4, 5],
            }
        )

        self.stocks = pd.DataFrame(
            {
                "ticker": ["MSFT", "GOOG", "AAPL"],
                "name": ["Microsoft Corporation", "Alphabet Inc", "Apple Inc"],
                "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
            }
        )

    def _create(self, schema, collection_name, metadata, engine, key):
        columns = []
        for col, col_type in schema.items():
            if col_type == int:
                col_type = db.Integer
            elif col_type == str:
                col_type = db.String
            elif col_type == datetime.datetime or col_type == pd.Timestamp:
                col_type = db.DateTime
            elif col_type == bool:
                col_type = db.Boolean
            elif col_type == float:
                col_type = db.Float
            else:
                raise TypeError(f"{col_type} unsupported type")
            columns.append(db.Column(col, col_type, primary_key=(col == key)))

        db.Table(collection_name, metadata, *columns)
        metadata.create_all(engine)
