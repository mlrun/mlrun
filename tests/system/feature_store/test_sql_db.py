import datetime

import pandas as pd
import pytest
from pymongo import MongoClient

import mlrun.feature_store as fs
from mlrun.datastore.sources import SqlDBSource, ParquetSource
from mlrun.datastore.targets import SqlDBTarget
from mlrun.feature_store.steps import FeaturesetValidator, MapClass, OneHotEncoder
from mlrun.features import MinMaxValidator
from tests.system.base import TestMLRunSystem
import sqlalchemy as db

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
        cls.db = env[
            "SQL_DB_PATH_STRING"
        ]
        cls.source_collection = "source_collection"
        cls.target_collection = "target_collection"

    def custom_setup(self):
        self._init_env_from_file()
        self.prepare_data()

    def get_data(self, data_name):
        if data_name == 'stocks':
            return self.stocks
        elif data_name == 'quotes':
            return self.quotes
        elif data_name == 'trades':
            return self.trades
        else:
            return None
    @staticmethod
    def get_schema(data_name):
        if data_name == 'stocks':
            return {"ticker": str, 'name': str, 'exchange': str}
        elif data_name == 'quotes':
            return {'time': pd.Timestamp, 'ticker': str,
                    'bid': float, 'ask': float, 'ind': int}
        elif data_name == 'trades':
            return {'time': pd.Timestamp, 'ticker': str,
                    'price': float, 'quantity': int, 'ind': int}
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

    @pytest.mark.parametrize("source_name, key", [('stocks', 'ticker'), ('trades', 'ind'),
                                                  ('quotes', 'ind')])
    def test_sql_source_basic(self, source_name, key):
        import sqlalchemy as db
        engine = db.create_engine(self.db, echo=True)
        sqlite_connection = engine.connect()
        origin_df = self.get_data(source_name)
        origin_df.to_sql(source_name, sqlite_connection, if_exists="replace")
        sqlite_connection.close()
        source = SqlDBSource(
            collection_name=source_name, db_path=self.db, key_field=key
        )

        feature_set = fs.FeatureSet(f'fs-{source_name}', entities=[fs.Entity(key)])
        df = fs.ingest(feature_set, source=source)
        df.reset_index(drop=True, inplace=True)
        origin_df.drop(columns=[key], inplace=True)
        origin_df.reset_index(drop=True, inplace=True)
        assert df.equals(origin_df)

    @pytest.mark.parametrize("source_name, key, encoder_col", [('stocks', 'ticker', 'exchange'),
                                                               ('trades', 'ind', 'ticker'),
                                                               ('quotes', 'ind', 'ticker')])
    def test_sql_source_with_step(self, source_name, key, encoder_col):
        import sqlalchemy as db
        engine = db.create_engine(self.db, echo=True)
        sqlite_connection = engine.connect()
        origin_df = self.get_data(source_name)
        origin_df.to_sql(source_name, sqlite_connection, if_exists="replace")
        sqlite_connection.close()

        # test source
        source = SqlDBSource(
            collection_name=source_name, db_path=self.db, key_field=key
        )
        feature_set = fs.FeatureSet(f'fs-{source_name}', entities=[fs.Entity(key)])
        one_hot_encoder_mapping = {
            encoder_col: list(origin_df[encoder_col].unique()),
        }
        feature_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
        df = fs.ingest(feature_set, source=source)

        # reference source
        feature_set_ref = fs.FeatureSet(f'fs-{source_name}-ref', entities=[fs.Entity(key)])
        feature_set_ref.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
        df_ref = fs.ingest(feature_set_ref, origin_df)

        assert df.equals(df_ref)

    @pytest.mark.parametrize("source_name, key, aggr_col", [('quotes', 'ind', 'ask'),
                                                            ('trades', 'ind', 'price')])
    def test_sql_source_with_aggregation(self, source_name, key, aggr_col):
        import sqlalchemy as db
        engine = db.create_engine(self.db, echo=True)
        sqlite_connection = engine.connect()
        origin_df = self.get_data(source_name)
        origin_df.to_sql(source_name, sqlite_connection, if_exists="replace")
        sqlite_connection.close()

        # test source
        source = SqlDBSource(
            collection_name=source_name, db_path=self.db, key_field=key
        )
        feature_set = fs.FeatureSet(f'fs-{source_name}', entities=[fs.Entity(key)])
        feature_set.add_aggregation(aggr_col, ["sum", "max"], "1h", "10m", name=f'{aggr_col}1')
        df = fs.ingest(feature_set, source=source)

        # reference source
        feature_set_ref = fs.FeatureSet(f'fs-{source_name}-ref', entities=[fs.Entity(key)])
        feature_set_ref.add_aggregation(aggr_col, ["sum", "max"], "1h", "10m", name=f'{aggr_col}1')
        df_ref = fs.ingest(feature_set_ref, origin_df)

        assert df.equals(df_ref)

    @pytest.mark.parametrize("target_name, key", [('stocks', 'ticker'),
                                                  ('trades', 'ind'),
                                                  ('quotes', 'ind')
                                                  ])
    def test_sql_target_basic(self, target_name, key):
        origin_df = self.get_data(target_name)
        schema = self.get_schema(target_name)

        target = SqlDBTarget(
            collection_name=target_name, db_path=self.db, create_collection=True,
            schema=schema, primary_key_column=key
        )
        feature_set = fs.FeatureSet(f'fs-{target_name}-tr', entities=[fs.Entity(key)])
        fs.ingest(feature_set, source=origin_df, targets=[target])
        df = target.as_df()

        origin_df.set_index(key, inplace=True)
        columns = [*schema.keys()]
        columns.remove(key)
        df.sort_index(inplace=True), origin_df.sort_index(inplace=True)

        assert df[columns].equals(origin_df[columns])

    @pytest.mark.parametrize("target_name, key", [
        # ('stocks', 'ticker'),
                                                  ('trades', 'ind'),
                                                  ('quotes', 'ind')
                                                  ])
    def test_sql_get_online_feature_basic(self, target_name, key):
        origin_df = self.get_data(target_name)
        schema = self.get_schema(target_name)

        target = SqlDBTarget(
            collection_name=target_name, db_path=self.db, create_collection=True,
            schema=schema, primary_key_column=key
        )
        feature_set = fs.FeatureSet(f'fs-{target_name}-tr', entities=[fs.Entity(key)])
        fs.ingest(feature_set, source=origin_df, targets=[target])
        columns = [*schema.keys()]
        columns.remove(key)
        features = [
            f'fs-{target_name}-tr.{columns[0]}',
            f'fs-{target_name}-tr.{columns[1]}'
        ]

        vector = fs.FeatureVector(
            f"{target_name}-vec", features, description="my test vector"
        )
        service = fs.get_online_feature_service(vector)
        print(service.get([{key: 1}], as_list=True))


    #
    # def test_mongodb_source_query(self):
    #     query_dict = {"author": {"$regex": "machine"}}
    #     source = MongoDBSource(
    #         connection_string=self.mongodb_connection_string,
    #         collection_name=self.collection,
    #         db_name=self.db,
    #         query=query_dict,
    #     )
    #     target = MongoDBTarget(
    #         connection_string=self.mongodb_connection_string,
    #         db_name=self.db,
    #         collection_name=self.target_collection,
    #         create_collection=True,
    #         override_collection=True,
    #     )
    #     self._test_mongodb_source_and_target(
    #         source, target, 10, self.cancellation_policy_val
    #     )
    #
    # def test_mongodb_source_query_with_chunk_size(self):
    #     query_dict = {"author": {"$regex": "machine"}}
    #     source = MongoDBSource(
    #         connection_string=self.mongodb_connection_string,
    #         collection_name=self.collection,
    #         db_name=self.db,
    #         query=query_dict,
    #         chunksize=10,
    #     )
    #     target = MongoDBTarget(
    #         connection_string=self.mongodb_connection_string,
    #         db_name=self.db,
    #         collection_name=self.target_collection,
    #         create_collection=True,
    #         override_collection=True,
    #     )
    #     self._test_mongodb_source_and_target(
    #         source, target, 10, self.cancellation_policy_val
    #     )
    #
    # def test_mongodb_source_query_with_time_filter(self):
    #     query_dict = {"author": {"$regex": "machine"}}
    #     start = datetime.datetime(2012, month=11, day=1, hour=0, minute=0, second=0)
    #     end = datetime.datetime(2012, month=12, day=1, hour=0, minute=0, second=0)
    #     source = MongoDBSource(
    #         connection_string=self.mongodb_connection_string,
    #         collection_name=self.collection,
    #         db_name=self.db,
    #         query=query_dict,
    #         start_time=start,
    #         end_time=end,
    #         time_field="date",
    #     )
    #     target = MongoDBTarget(
    #         connection_string=self.mongodb_connection_string,
    #         db_name=self.db,
    #         collection_name=self.target_collection,
    #         create_collection=True,
    #         override_collection=True,
    #     )
    #     self._test_mongodb_source_and_target(
    #         source, target, 10, self.cancellation_policy_val
    #     )
    #
    # def test_sql_source(self):
    #     self.prepare_data()
    #     DB_S = "my_dataset"
    #     stocks_set = fs.FeatureSet("stocks", entities=[fs.Entity("ticker")])
    #     stocks_target = MongoDBTarget(
    #         connection_string=self.mongodb_connection_string,
    #         db_name=DB_S,
    #         collection_name="stocks_fs",
    #         create_collection=True,
    #         override_collection=True,
    #     )
    #
    #     stocks_set.set_targets(targets=[stocks_target])
    #     stocks_set.plot(rankdir="LR", with_targets=True)
    #     fs.ingest(
    #         stocks_set,
    #         self.stocks,
    #         infer_options=fs.InferOptions.default(),
    #         targets=[stocks_target],
    #     )
    #
    #     quotes_set = fs.FeatureSet("stock-quotes", entities=[fs.Entity("ticker")])
    #     quotes_set.graph.to("MyMap", multiplier=3).to(
    #         "storey.Extend", _fn="({'extra': event['bid'] * 77})"
    #     ).to("storey.Filter", "filter", _fn="(event['bid'] > 51.92)").to(
    #         FeaturesetValidator()
    #     )
    #
    #     quotes_set.add_aggregation("ask", ["sum", "max"], "1h", "10m", name="asks1")
    #     quotes_set.add_aggregation("ask", ["sum", "max"], "5h", "10m", name="asks5")
    #     quotes_set.add_aggregation("bid", ["min", "max"], "1h", "10m", name="bids")
    #
    #     quotes_set["bid"] = fstore.Feature(
    #         validator=MinMaxValidator(min=52, severity="info"),
    #         value_type=fstore.ValueType.FLOAT,
    #     )
    #     quotes_target = MongoDBTarget(
    #         connection_string=self.mongodb_connection_string,
    #         db_name=DB_S,
    #         collection_name="quotes_fs",
    #         create_collection=True,
    #         override_collection=True,
    #     )
    #
    #     # add default target definitions and plot
    #     quotes_set.set_targets(targets=[quotes_target])
    #     fstore.ingest(
    #         quotes_set,
    #         self.quotes,
    #         targets=[quotes_target],
    #     )
    #
    #     features = [
    #         "stock-quotes.multi",
    #         "stock-quotes.asks5_sum_5h as total_ask",
    #         "stock-quotes.bids_min_1h",
    #         "stock-quotes.bids_max_1h",
    #         "stocks.*",
    #     ]
    #
    #     vector = fstore.FeatureVector(
    #         "stocks-vec", features, description="stocks demo feature vector"
    #     )
    #     vector.save()
    #     service = fs.get_online_feature_service("stocks-vec")
    #     assert service.get([{"ticker": "AAPL"}], as_list=True) == [
    #         [293.96999999999997, 98.01, 97.99, 97.99, "Apple Inc", "NASDAQ"]
    #     ]
    #     assert service.get([{"ticker": "AAPL"}],) == [
    #         {
    #             "bids_min_1h": 97.99,
    #             "bids_max_1h": 97.99,
    #             "total_ask": 98.01,
    #             "multi": 293.96999999999997,
    #             "name": "Apple Inc",
    #             "exchange": "NASDAQ",
    #         }
    #     ]
    #     assert service.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}], as_list=True) == [
    #         [2161.5, 2162.74, 720.5, 720.5, "Alphabet Inc", "NASDAQ"],
    #         [156.03, 207.97, 51.95, 52.01, "Microsoft Corporation", "NASDAQ"],
    #     ]
    #     assert service.get([{"ticker": "GOOG"}, {"ticker": "MSFT"}],) == [
    #         {
    #             "bids_min_1h": 720.5,
    #             "bids_max_1h": 720.5,
    #             "total_ask": 2162.74,
    #             "multi": 2161.5,
    #             "name": "Alphabet Inc",
    #             "exchange": "NASDAQ",
    #         },
    #         {
    #             "bids_min_1h": 51.95,
    #             "bids_max_1h": 52.01,
    #             "total_ask": 207.97,
    #             "multi": 156.03,
    #             "name": "Microsoft Corporation",
    #             "exchange": "NASDAQ",
    #         },
    #     ]

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

#     @staticmethod
#     def _test_mongodb_source_and_target(
#         source: MongoDBSource, target: MongoDBTarget, column_amount: int, map: list
#     ):
#         targets = [target]
#         feature_set = fs.FeatureSet(
#             "sample_training_posts",
#             entities=[fs.Entity("_id")],
#             description="feature set",
#         )
#
#         one_hot_encoder_mapping = {
#             "title": list(map),
#         }
#         feature_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))
#
#         ingest_df = fs.ingest(
#             feature_set,
#             source=source,
#             targets=targets,
#             infer_options=fs.InferOptions.default(),
#         )
#
#         assert ingest_df is not None
#         assert len(ingest_df.columns) == column_amount
#         assert len(target.as_df().columns) == column_amount + 2
#
#
# class MyMap(MapClass):
#     def __init__(self, multiplier=1, **kwargs):
#         super().__init__(**kwargs)
#         self._multiplier = multiplier
#
#     def do(self, event):
#         event["multi"] = event["bid"] * self._multiplier
#         return event
