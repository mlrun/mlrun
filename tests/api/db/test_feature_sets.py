import pytest
from sqlalchemy.orm import Session

from mlrun.api.db.base import DBInterface
from tests.api.db.conftest import dbs


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_create_feature_set(db: DBInterface, db_session: Session):
    fs = {
        "metadata": {"name": "dummy", "labels": {"owner": "saarc", "group": "dev"}},
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

    proj = "proj_test"
    name = fs["metadata"]["name"]

    fs_id = db.add_feature_set(db_session, proj, fs, versioned=True)
    print("Got ID: {}".format(fs_id))

    fs_res = db.get_feature_set(db_session, proj, name)
    print("Got from DB: {}".format(fs_res.dict()))

    fs_res = db.list_feature_sets(db_session, proj)
