import pytest
from sqlalchemy.orm import Session
from mlrun.artifacts.plots import ChartArtifact, PlotArtifact

from mlrun.api.db.base import DBInterface
from tests.api.db.conftest import dbs


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_artifact_name_filter(db: DBInterface, db_session: Session):
    artifact_name_1 = "artifact_name_1"
    artifact_name_2 = "artifact_name_2"
    artifact_1 = {"metadata": {"name": artifact_name_1}, "status": {"bla": "blabla"}}
    artifact_2 = {"metadata": {"name": artifact_name_2}, "status": {"bla": "blabla"}}
    uid = "artifact_uid"

    db.store_artifact(
        db_session, artifact_name_1, artifact_1, uid,
    )
    db.store_artifact(
        db_session, artifact_name_2, artifact_2, uid,
    )
    artifacts = db.list_artifacts(db_session)
    assert len(artifacts) == 2

    artifacts = db.list_artifacts(db_session, name=artifact_name_1)
    assert len(artifacts) == 1
    assert artifacts[0]["metadata"]["name"] == artifact_name_1

    artifacts = db.list_artifacts(db_session, name=artifact_name_2)
    assert len(artifacts) == 1
    assert artifacts[0]["metadata"]["name"] == artifact_name_2

    artifacts = db.list_artifacts(db_session, name="artifact_name")
    assert len(artifacts) == 2


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_artifact_kind_filter(db: DBInterface, db_session: Session):
    artifact_name_1 = "artifact_name_1"
    artifact_kind_1 = ChartArtifact.kind
    artifact_name_2 = "artifact_name_2"
    artifact_kind_2 = PlotArtifact.kind
    artifact_1 = {"metadata": {"name": artifact_name_1}, 'kind': artifact_kind_1, "status": {"bla": "blabla"}}
    artifact_2 = {"metadata": {"name": artifact_name_2}, 'kind': artifact_kind_2, "status": {"bla": "blabla"}}
    uid = "artifact_uid"

    db.store_artifact(
        db_session, artifact_name_1, artifact_1, uid,
    )
    db.store_artifact(
        db_session, artifact_name_2, artifact_2, uid,
    )
    artifacts = db.list_artifacts(db_session)
    assert len(artifacts) == 2

    artifacts = db.list_artifacts(db_session, kind=artifact_kind_1)
    assert len(artifacts) == 1
    assert artifacts[0]["metadata"]["name"] == artifact_name_1

    artifacts = db.list_artifacts(db_session, kind=artifact_kind_2)
    assert len(artifacts) == 1
    assert artifacts[0]["metadata"]["name"] == artifact_name_2
