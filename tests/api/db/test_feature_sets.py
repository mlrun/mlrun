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
#
import deepdiff
import pytest
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.db.base import DBInterface
from tests.api.db.conftest import dbs


def _create_feature_set(name):
    return {
        "metadata": {"name": name, "labels": {"owner": "saarc", "group": "dev"}},
        "spec": {
            "entities": [{"name": "ticker", "value_type": "str"}],
            "features": [
                {"name": "time", "value_type": "datetime"},
                {"name": "bid", "value_type": "float"},
                {"name": "ask", "value_type": "time"},
            ],
        },
        "status": {
            "state": "created",
            "stats": {
                "time": {
                    "count": "8",
                    "unique": "7",
                    "top": "2016-05-25 13:30:00.222222",
                }
            },
        },
    }


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_create_feature_set(db: DBInterface, db_session: Session):
    name = "dummy"
    feature_set = _create_feature_set(name)

    project = "proj-test"

    feature_set = schemas.FeatureSet(**feature_set)
    db.store_feature_set(
        db_session, project, name, feature_set, tag="latest", versioned=True
    )
    db.get_feature_set(db_session, project, name)

    feature_set_res = db.list_feature_sets(db_session, project)
    assert len(feature_set_res.feature_sets) == 1

    features_res = db.list_features(db_session, project, "time")
    assert len(features_res.features) == 1


@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_update_feature_set(db: DBInterface, db_session: Session):
    name = "dummy"
    feature_set = _create_feature_set(name)

    project = "proj-test"

    feature_set = schemas.FeatureSet(**feature_set)
    db.store_feature_set(
        db_session, project, name, feature_set, tag="latest", versioned=True
    )
    assert feature_set.metadata.labels != {}

    feature_set = db.get_feature_set(db_session, project, name)
    old_feature_set_dict = feature_set.dict()

    # remove labels from feature set and store it
    feature_set.metadata.labels = {}
    db.store_feature_set(
        db_session, project, name, feature_set, tag="latest", versioned=True
    )
    updated_feature_set = db.get_feature_set(db_session, project, name)
    assert updated_feature_set.metadata.labels == {}

    old_feature_set_dict["metadata"]["labels"] = {}
    updated_feature_set_dict = updated_feature_set.dict()

    # uid and update time change due to rehashing and the store operation
    assert (
        old_feature_set_dict["metadata"]["uid"]
        != updated_feature_set_dict["metadata"]["uid"]
    )
    assert (
        old_feature_set_dict["metadata"]["updated"]
        != updated_feature_set_dict["metadata"]["updated"]
    )
    old_feature_set_dict["metadata"].pop("uid")
    old_feature_set_dict["metadata"].pop("updated")
    updated_feature_set_dict["metadata"].pop("uid")
    updated_feature_set_dict["metadata"].pop("updated")

    assert deepdiff.DeepDiff(old_feature_set_dict, updated_feature_set_dict) == {}
