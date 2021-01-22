import deepdiff
import numpy
import pandas
import pytest
from sqlalchemy.orm import Session
import mlrun.artifacts.dataset
from mlrun.artifacts.plots import ChartArtifact, PlotArtifact
from mlrun.artifacts.dataset import DatasetArtifact
from mlrun.artifacts.model import ModelArtifact

from mlrun.api import schemas
from mlrun.api.db.base import DBInterface
from tests.api.db.conftest import dbs


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
    artifact_1 = _generate_artifact(artifact_name_1, artifact_kind_1)
    artifact_2 = _generate_artifact(artifact_name_2, artifact_kind_2)
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
    artifact_1 = _generate_artifact(artifact_name_1, artifact_kind_1)
    artifact_2 = _generate_artifact(artifact_name_2, artifact_kind_2)
    artifact_3 = _generate_artifact(artifact_name_3, artifact_kind_3)
    artifact_4 = _generate_artifact(artifact_name_4, artifact_kind_4)
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
def test_data_migration_fix_datasets_large_previews(
    db: DBInterface, db_session: Session,
):
    artifact_with_valid_preview_key = "artifact-with-valid-preview-key"
    artifact_with_valid_preview_uid = "artifact-with-valid-preview-uid"
    artifact_with_valid_preview = mlrun.artifacts.DatasetArtifact(
        artifact_with_valid_preview_key,
        df=pandas.DataFrame(
            [{"A": 10, "B": 100}, {"A": 11, "B": 110}, {"A": 12, "B": 120}]
        ),
    )
    db.store_artifact(
        db_session,
        artifact_with_valid_preview_key,
        artifact_with_valid_preview.to_dict(),
        artifact_with_valid_preview_uid,
    )

    artifact_with_invalid_preview_key = "artifact-with-invalid-preview-key"
    artifact_with_invalid_preview_uid = "artifact-with-invalid-preview-uid"
    artifact_with_invalid_preview = mlrun.artifacts.DatasetArtifact(
        artifact_with_invalid_preview_key,
        df=pandas.DataFrame(
            numpy.random.randint(
                0, 10, size=(10, mlrun.artifacts.dataset.max_preview_columns * 3)
            )
        ),
        ignore_preview_limits=True,
    )
    db.store_artifact(
        db_session,
        artifact_with_invalid_preview_key,
        artifact_with_invalid_preview.to_dict(),
        artifact_with_invalid_preview_uid,
    )

    # perform the migration
    mlrun.api.initial_data._fix_datasets_large_previews(db, db_session)

    artifact_with_valid_preview_after_migration = db.read_artifact(
        db_session, artifact_with_valid_preview_key, artifact_with_valid_preview_uid
    )
    assert (
        deepdiff.DeepDiff(
            artifact_with_valid_preview_after_migration,
            artifact_with_valid_preview.to_dict(),
            ignore_order=True,
            exclude_paths=["root['updated']"],
        )
        == {}
    )

    artifact_with_invalid_preview_after_migration = db.read_artifact(
        db_session, artifact_with_invalid_preview_key, artifact_with_invalid_preview_uid
    )
    assert (
        deepdiff.DeepDiff(
            artifact_with_invalid_preview_after_migration,
            artifact_with_invalid_preview.to_dict(),
            ignore_order=True,
            exclude_paths=[
                "root['updated']",
                "root['header']",
                "root['stats']",
                "root['schema']",
                "root['preview']",
            ],
        )
        == {}
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["header"])
        == mlrun.artifacts.dataset.max_preview_columns
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["stats"])
        == mlrun.artifacts.dataset.max_preview_columns - 1
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["preview"][0])
        == mlrun.artifacts.dataset.max_preview_columns
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["schema"]["fields"])
        == mlrun.artifacts.dataset.max_preview_columns + 1
    )


def _generate_artifact(name, kind=None):
    artifact = {
        "metadata": {"name": name},
        "kind": kind,
        "status": {"bla": "blabla"},
    }
    if kind:
        artifact["kind"] = kind

    return artifact
