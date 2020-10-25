import pytest
from sqlalchemy.orm import Session

from mlrun.api.db.base import DBInterface
from tests.api.db.conftest import dbs
from mlrun.api.db.sqldb.models import FeatureSet


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_create_feature_set(db: DBInterface, db_session: Session):
    fs = {"name": "fs1",
          "project": "proj1",
          "entities": [
              {
                  "name": "ticker",
                  "value_type": "str",
              },
          ],
          "features": [
              {
                  "name": "time",
                  "value_type": "datetime",
              },
              {
                  "name": "bid",
                  "value_type": "float",
              },
              {
                  "name": "ask",
                  "value_type": "time",
              },
          ],
          "status": {
              "state": "created",
              "stats": {
                  "time": {
                      "count": "8",
                      "unique": "7",
                      "top": "2016-05-25 13:30:00.222222"
                  }
              },
          },

          }

    fs_id = db.add_feature_set(db_session, fs)
    print("Got ID: {}".format(fs_id))

    fs_res = db.get_feature_set(db_session, fs["name"], fs["project"])
    print("Got from DB: {}".format(fs_res))

