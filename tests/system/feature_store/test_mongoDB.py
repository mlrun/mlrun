import datetime

import pandas as pd
import pytest
from pymongo import MongoClient

import mlrun.feature_store as fs
from mlrun.datastore.sources import MongoDBSource
from mlrun.datastore.targets import MongoDBTarget
from mlrun.feature_store.steps import OneHotEncoder
from tests.system.base import TestMLRunSystem

CREDENTIALS_ENV = "MLRUN_SYSTEM_TESTS_MONGODB_CONNECTION_STRING"


def _are_mongodb_connection_string_not_set() -> bool:

    return True


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.skipif(
    _are_mongodb_connection_string_not_set(),
    reason=f"Environment variable {CREDENTIALS_ENV} is not defined",
)
@pytest.mark.enterprise
class TestFeatureStoreMongoDB(TestMLRunSystem):
    project_name = "fs-system-test-mongodb"

    @classmethod
    def _init_env_from_file(cls):
        env = cls._get_env_from_file()
        cls.mongodb_connection_string = env[
            "MLRUN_SYSTEM_TESTS_MONGODB_CONNECTION_STRING"
        ]
        cls.db = "sample_training"
        cls.collection = "posts"
        cls.target_collection = "posts_after_fs"

    def custom_setup(self):
        self._init_env_from_file()
        client = MongoClient(self.mongodb_connection_string)
        mydatabase = client[self.db]
        mycollection = mydatabase[self.collection]
        df = pd.DataFrame(list(mycollection.find()))
        self.cancellation_policy_val = df["title"].unique()
        # self.cancellation_policy_val = ['Bill of Rights', 'US Constitution', 'Gettysburg Address',
        #                                 'Declaration of Independence']

    def test_mongodb_source_query(self):
        query_dict = {"author": {"$regex": "machine"}}
        source = MongoDBSource(
            connection_string=self.mongodb_connection_string,
            collection_name=self.collection,
            db_name=self.db,
            query=query_dict,
        )
        target = MongoDBTarget(
            connection_string=self.mongodb_connection_string,
            db_name=self.db,
            collection_name=self.target_collection,
            create_collection=True,
            override_collection=True,
        )
        self._test_mongodb_source(source, target, 10, self.cancellation_policy_val)

    def test_mongodb_source_query_with_chunk_size(self):
        query_dict = {"author": {"$regex": "machine"}}
        source = MongoDBSource(
            connection_string=self.mongodb_connection_string,
            collection_name=self.collection,
            db_name=self.db,
            query=query_dict,
            chunksize=10,
        )
        target = MongoDBTarget(
            connection_string=self.mongodb_connection_string,
            db_name=self.db,
            collection_name=self.target_collection,
            create_collection=True,
            override_collection=True,
        )
        self._test_mongodb_source(source, target, 10, self.cancellation_policy_val)

    def test_mongodb_source_query_with_time_filter(self):
        query_dict = {"author": {"$regex": "machine"}}
        start = datetime.datetime(2012, month=11, day=1, hour=0, minute=0, second=0)
        end = datetime.datetime(2012, month=12, day=1, hour=0, minute=0, second=0)
        source = MongoDBSource(
            connection_string=self.mongodb_connection_string,
            collection_name=self.collection,
            db_name=self.db,
            query=query_dict,
            start_time=start,
            end_time=end,
            time_field="date",
        )
        target = MongoDBTarget(
            connection_string=self.mongodb_connection_string,
            db_name=self.db,
            collection_name=self.target_collection,
            create_collection=True,
            override_collection=True,
        )
        self._test_mongodb_source(source, target, 10, self.cancellation_policy_val)

    @staticmethod
    def _test_mongodb_source(
        source: MongoDBSource, target: MongoDBTarget, column_amount: int, map: list
    ):
        targets = [target]
        feature_set = fs.FeatureSet(
            "sample_training_posts",
            entities=[fs.Entity("_id")],
            description="feature set",
        )

        one_hot_encoder_mapping = {
            "title": list(map),
        }
        feature_set.graph.to(OneHotEncoder(mapping=one_hot_encoder_mapping))

        ingest_df = fs.ingest(
            feature_set,
            source=source,
            targets=targets,
            infer_options=fs.InferOptions.default(),
        )

        assert ingest_df is not None
        assert len(ingest_df.columns) == column_amount
        assert len(target.as_df().columns) == column_amount + 1
