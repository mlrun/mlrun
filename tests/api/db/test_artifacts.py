import pytest
import deepdiff
from sqlalchemy.orm import Session
from mlrun.artifacts.plots import ChartArtifact, PlotArtifact
from mlrun.artifacts.dataset import DatasetArtifact
from mlrun.artifacts.model import ModelArtifact

import mlrun.api.initial_data
from mlrun.api import schemas
from mlrun.api.db.base import DBInterface
from tests.api.db.conftest import dbs
from mlrun.utils import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import MultipleResultsFound
from mlrun.api.db.base import DBError


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_artifact_name_filter(db: DBInterface, db_session: Session):
    artifact_name_1 = "artifact_name_1"
    artifact_name_2 = "artifact_name_2"
    artifact_1 = _generate_artifact(artifact_name_1)
    artifact_2 = _generate_artifact(artifact_name_2)
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
    artifact_1 = _generate_artifact(artifact_name_1, kind=artifact_kind_1)
    artifact_2 = _generate_artifact(artifact_name_2, kind=artifact_kind_2)
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


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_artifact_category_filter(db: DBInterface, db_session: Session):
    artifact_name_1 = "artifact_name_1"
    artifact_kind_1 = ChartArtifact.kind
    artifact_name_2 = "artifact_name_2"
    artifact_kind_2 = PlotArtifact.kind
    artifact_name_3 = "artifact_name_3"
    artifact_kind_3 = ModelArtifact.kind
    artifact_name_4 = "artifact_name_4"
    artifact_kind_4 = DatasetArtifact.kind
    artifact_1 = _generate_artifact(artifact_name_1, kind=artifact_kind_1)
    artifact_2 = _generate_artifact(artifact_name_2, kind=artifact_kind_2)
    artifact_3 = _generate_artifact(artifact_name_3, kind=artifact_kind_3)
    artifact_4 = _generate_artifact(artifact_name_4, kind=artifact_kind_4)
    uid = "artifact_uid"

    db.store_artifact(
        db_session, artifact_name_1, artifact_1, uid,
    )
    db.store_artifact(
        db_session, artifact_name_2, artifact_2, uid,
    )
    db.store_artifact(
        db_session, artifact_name_3, artifact_3, uid,
    )
    db.store_artifact(
        db_session, artifact_name_4, artifact_4, uid,
    )
    artifacts = db.list_artifacts(db_session)
    assert len(artifacts) == 4

    artifacts = db.list_artifacts(db_session, category=schemas.ArtifactCategories.model)
    assert len(artifacts) == 1
    assert artifacts[0]["metadata"]["name"] == artifact_name_3

    artifacts = db.list_artifacts(
        db_session, category=schemas.ArtifactCategories.dataset
    )
    assert len(artifacts) == 1
    assert artifacts[0]["metadata"]["name"] == artifact_name_4

    artifacts = db.list_artifacts(db_session, category=schemas.ArtifactCategories.other)
    assert len(artifacts) == 2
    assert artifacts[0]["metadata"]["name"] == artifact_name_1
    assert artifacts[1]["metadata"]["name"] == artifact_name_2


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_artifact_tagging(db: DBInterface, db_session: Session):
    artifact_1_key = "artifact_key_1"
    artifact_1_body = _generate_artifact(artifact_1_key)
    artifact_1_kind = ChartArtifact.kind
    artifact_1_with_kind_body = _generate_artifact(artifact_1_key, kind=artifact_1_kind)
    artifact_1_uid = "artifact_uid"
    artifact_1_with_kind_uid = "artifact_uid_2"

    db.store_artifact(
        db_session, artifact_1_key, artifact_1_body, artifact_1_uid,
    )
    db.store_artifact(
        db_session, artifact_1_key, artifact_1_with_kind_body, artifact_1_with_kind_uid,
    )
    artifact = db.read_artifact(db_session, artifact_1_key, tag="latest")
    assert artifact["kind"] == artifact_1_kind
    artifact = db.read_artifact(db_session, artifact_1_key, tag=artifact_1_uid)
    assert artifact.get("kind") is None


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_data_migration_fix_artifact_tags_duplications(
    db: DBInterface, db_session: Session,
):
    def _buggy_tag_artifacts(session, objs, project: str, name: str):
        # This is the function code that was used before we did the fix and added the data migration
        for obj in objs:
            tag = obj.Tag(project=project, name=name, obj_id=obj.id)
            _upsert(session, tag, ignore=True)

    def _upsert(session, obj, ignore=False):
        try:
            session.add(obj)
            session.commit()
        except SQLAlchemyError as err:
            session.rollback()
            cls = obj.__class__.__name__
            logger.warning(f"conflict adding {cls}, {err}")
            if not ignore:
                raise DBError(f"duplicate {cls} - {err}") from err

    db.tag_artifacts = _buggy_tag_artifacts

    artifact_1_key = "artifact_key_1"
    artifact_1_uid = "artifact_1_uid_1"
    artifact_1_body = _generate_artifact(artifact_1_key, artifact_1_uid)
    artifact_1_kind = ChartArtifact.kind
    artifact_1_with_kind_uid = "artifact_1_uid_2"
    artifact_1_with_kind_body = _generate_artifact(
        artifact_1_key, artifact_1_with_kind_uid, kind=artifact_1_kind
    )
    artifact_2_key = "artifact_key_2"
    artifact_2_uid = "artifact_2_uid_1"
    artifact_2_body = _generate_artifact(artifact_2_key, artifact_2_uid)
    artifact_2_kind = PlotArtifact.kind
    artifact_2_with_kind_uid = "artifact_2_uid_2"
    artifact_2_with_kind_body = _generate_artifact(
        artifact_2_key, artifact_2_with_kind_uid, kind=artifact_2_kind
    )
    artifact_3_key = "artifact_key_3"
    artifact_3_kind = DatasetArtifact.kind
    artifact_3_with_kind_uid = "artifact_3_uid_1"
    artifact_3_with_kind_body = _generate_artifact(
        artifact_3_key, artifact_3_with_kind_uid, kind=artifact_3_kind
    )

    db.store_artifact(
        db_session, artifact_1_key, artifact_1_body, artifact_1_uid,
    )
    db.store_artifact(
        db_session, artifact_1_key, artifact_1_with_kind_body, artifact_1_with_kind_uid,
    )
    db.store_artifact(
        db_session, artifact_2_key, artifact_2_body, artifact_2_uid, tag="not-latest"
    )
    db.store_artifact(
        db_session,
        artifact_2_key,
        artifact_2_with_kind_body,
        artifact_2_with_kind_uid,
        tag="not-latest",
    )
    db.store_artifact(
        db_session, artifact_3_key, artifact_3_with_kind_body, artifact_3_with_kind_uid
    )

    # Before the migration:
    # 1. read artifact would have failed when there's more than one tag record with the same key (happen when you
    # store twice)
    with pytest.raises(MultipleResultsFound):
        db.read_artifact(db_session, artifact_1_key, tag="latest")
    with pytest.raises(MultipleResultsFound):
        db.read_artifact(db_session, artifact_2_key, tag="not-latest")

    # 2. read artifact would have succeed when there's only one tag record with the same key (happen when you
    # stored only once)
    artifact = db.read_artifact(db_session, artifact_3_key, tag="latest")
    assert artifact["metadata"]["uid"] == artifact_3_with_kind_uid

    # 3. list artifact without tag would have returned the latest (by update time) of each artifact key
    artifacts = db.list_artifacts(db_session)
    assert len(artifacts) == len([artifact_1_key, artifact_2_key, artifact_3_key])
    assert (
        deepdiff.DeepDiff(
            [artifact["metadata"]["uid"] for artifact in artifacts],
            [
                artifact_1_with_kind_uid,
                artifact_2_with_kind_uid,
                artifact_3_with_kind_uid,
            ],
            ignore_order=True,
        )
        == {}
    )

    # 4. list artifact with tag would have returned all of the artifact that at some point were tagged with the given
    # tag
    artifacts = db.list_artifacts(db_session, tag="latest")
    assert len(artifacts) == len(
        [artifact_1_uid, artifact_1_with_kind_uid, artifact_3_with_kind_uid]
    )

    # perform the migration
    mlrun.api.initial_data._fix_artifact_tags_duplications(db, db_session)

    # After the migration:
    # 1. read artifact should succeed (fixed) and return the latest updated record that was tagged with the requested
    # tag
    artifact = db.read_artifact(db_session, artifact_1_key, tag="latest")
    assert artifact["metadata"]["uid"] == artifact_1_with_kind_uid
    artifact = db.read_artifact(db_session, artifact_2_key, tag="not-latest")
    assert artifact["metadata"]["uid"] == artifact_2_with_kind_uid

    # 2. read artifact should (still) succeed when there's only one tag record with the same key (happen when you
    # stored only once)
    artifact = db.read_artifact(db_session, artifact_3_key, tag="latest")
    assert artifact["metadata"]["uid"] == artifact_3_with_kind_uid

    # 3. list artifact without tag should (still) return the latest (by update time) of each artifact key
    artifacts = db.list_artifacts(db_session)
    assert len(artifacts) == len([artifact_1_key, artifact_2_key, artifact_3_key])
    assert (
        deepdiff.DeepDiff(
            [artifact["metadata"]["uid"] for artifact in artifacts],
            [
                artifact_1_with_kind_uid,
                artifact_2_with_kind_uid,
                artifact_3_with_kind_uid,
            ],
            ignore_order=True,
        )
        == {}
    )

    # 4. list artifact with tag should (fixed) return all of the artifact that are tagged with the given tag
    artifacts = db.list_artifacts(db_session, tag="latest")
    assert (
        deepdiff.DeepDiff(
            [artifact["metadata"]["uid"] for artifact in artifacts],
            [artifact_1_with_kind_uid, artifact_3_with_kind_uid],
            ignore_order=True,
        )
        == {}
    )


def _generate_artifact(name, uid=None, kind=None):
    artifact = {
        "metadata": {"name": name},
        "kind": kind,
        "status": {"bla": "blabla"},
    }
    if kind:
        artifact["kind"] = kind
    if uid:
        artifact["metadata"]["uid"] = uid

    return artifact
