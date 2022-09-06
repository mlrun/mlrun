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
import numpy
import pandas
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import MultipleResultsFound

import mlrun.api.initial_data
import mlrun.errors
from mlrun.api import schemas
from mlrun.api.db.base import DBError, DBInterface
from mlrun.api.schemas.artifact import ArtifactCategories
from mlrun.artifacts.dataset import DatasetArtifact
from mlrun.artifacts.model import ModelArtifact
from mlrun.artifacts.plots import ChartArtifact, PlotArtifact
from mlrun.utils import logger
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
        db_session,
        artifact_name_1,
        artifact_1,
        uid,
    )
    db.store_artifact(
        db_session,
        artifact_name_2,
        artifact_2,
        uid,
    )
    artifacts = db.list_artifacts(db_session)
    assert len(artifacts) == 2

    artifacts = db.list_artifacts(db_session, name=artifact_name_1)
    assert len(artifacts) == 1
    assert artifacts[0]["metadata"]["name"] == artifact_name_1

    artifacts = db.list_artifacts(db_session, name=artifact_name_2)
    assert len(artifacts) == 1
    assert artifacts[0]["metadata"]["name"] == artifact_name_2

    artifacts = db.list_artifacts(db_session, name="~artifact_name")
    assert len(artifacts) == 2


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_artifact_iter_parameter(db: DBInterface, db_session: Session):
    artifact_name_1 = "artifact_name_1"
    artifact_name_2 = "artifact_name_2"
    artifact_1 = _generate_artifact(artifact_name_1)
    artifact_2 = _generate_artifact(artifact_name_2)
    uid = "artifact_uid"

    # Use iters with multiple digits, to make sure filtering them via regex works
    test_iters = [0, 5, 9, 42, 219, 2102]
    for iter in test_iters:
        artifact_1["iter"] = artifact_2["iter"] = iter
        db.store_artifact(db_session, artifact_name_1, artifact_1, uid, iter)
        db.store_artifact(db_session, artifact_name_2, artifact_2, uid, iter)

    # No filter on iter. All are expected
    artifacts = db.list_artifacts(db_session)
    assert len(artifacts) == len(test_iters) * 2

    # Look for the various iteration numbers. Note that 0 is a special case due to the DB structure
    for iter in test_iters:
        artifacts = db.list_artifacts(db_session, iter=iter)
        assert len(artifacts) == 2
        for artifact in artifacts:
            assert artifact["iter"] == iter

    # Negative test
    artifacts = db.list_artifacts(db_session, iter=666)
    assert len(artifacts) == 0

    # Iter filter and a name filter, make sure query composition works
    artifacts = db.list_artifacts(db_session, name=artifact_name_1, iter=2102)
    assert len(artifacts) == 1


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
        db_session,
        artifact_name_1,
        artifact_1,
        uid,
    )
    db.store_artifact(
        db_session,
        artifact_name_2,
        artifact_2,
        uid,
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
        db_session,
        artifact_name_1,
        artifact_1,
        uid,
    )
    db.store_artifact(
        db_session,
        artifact_name_2,
        artifact_2,
        uid,
    )
    db.store_artifact(
        db_session,
        artifact_name_3,
        artifact_3,
        uid,
    )
    db.store_artifact(
        db_session,
        artifact_name_4,
        artifact_4,
        uid,
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
        db_session,
        artifact_1_key,
        artifact_1_body,
        artifact_1_uid,
    )
    db.store_artifact(
        db_session,
        artifact_1_key,
        artifact_1_with_kind_body,
        artifact_1_with_kind_uid,
    )
    artifact = db.read_artifact(db_session, artifact_1_key, tag="latest")
    assert artifact["kind"] == artifact_1_kind
    artifact = db.read_artifact(db_session, artifact_1_key, tag=artifact_1_uid)
    assert artifact.get("kind") is None
    artifacts = db.list_artifacts(db_session, artifact_1_key, tag="latest")
    assert len(artifacts) == 1
    artifacts = db.list_artifacts(db_session, artifact_1_key, tag=artifact_1_uid)
    assert len(artifacts) == 1


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_store_artifact_restoring_multiple_tags(db: DBInterface, db_session: Session):
    artifact_key = "artifact_key_1"
    artifact_1_uid = "artifact_uid_1"
    artifact_2_uid = "artifact_uid_2"
    artifact_1_body = _generate_artifact(artifact_key, uid=artifact_1_uid)
    artifact_2_body = _generate_artifact(artifact_key, uid=artifact_2_uid)
    artifact_1_tag = "artifact_tag_1"
    artifact_2_tag = "artifact_tag_2"

    db.store_artifact(
        db_session,
        artifact_key,
        artifact_1_body,
        artifact_1_uid,
        tag=artifact_1_tag,
    )
    db.store_artifact(
        db_session,
        artifact_key,
        artifact_2_body,
        artifact_2_uid,
        tag=artifact_2_tag,
    )
    artifacts = db.list_artifacts(db_session, artifact_key, tag="*")
    assert len(artifacts) == 2
    expected_uids = [artifact_1_uid, artifact_2_uid]
    uids = [artifact["metadata"]["uid"] for artifact in artifacts]
    assert (
        deepdiff.DeepDiff(
            expected_uids,
            uids,
            ignore_order=True,
        )
        == {}
    )
    expected_tags = [artifact_1_tag, artifact_2_tag]
    tags = [artifact["metadata"]["tag"] for artifact in artifacts]
    assert (
        deepdiff.DeepDiff(
            expected_tags,
            tags,
            ignore_order=True,
        )
        == {}
    )
    artifact = db.read_artifact(db_session, artifact_key, tag=artifact_1_tag)
    assert artifact["metadata"]["uid"] == artifact_1_uid
    assert artifact["metadata"]["tag"] == artifact_1_tag
    artifact = db.read_artifact(db_session, artifact_key, tag=artifact_2_tag)
    assert artifact["metadata"]["uid"] == artifact_2_uid
    assert artifact["metadata"]["tag"] == artifact_2_tag


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_read_artifact_tag_resolution(db: DBInterface, db_session: Session):
    """
    We had a bug in which when we got a tag filter for read/list artifact, we were transforming this tag to list of
    possible uids which is wrong, since a different artifact might have this uid as well, and we will return it,
    although it's not really tag with the given tag
    """
    artifact_1_key = "artifact_key_1"
    artifact_2_key = "artifact_key_2"
    artifact_uid = "artifact_uid_1"
    artifact_1_body = _generate_artifact(artifact_1_key, uid=artifact_uid)
    artifact_2_body = _generate_artifact(artifact_2_key, uid=artifact_uid)
    artifact_1_tag = "artifact_tag_1"
    artifact_2_tag = "artifact_tag_2"

    db.store_artifact(
        db_session,
        artifact_1_key,
        artifact_1_body,
        artifact_uid,
        tag=artifact_1_tag,
    )
    db.store_artifact(
        db_session,
        artifact_2_key,
        artifact_2_body,
        artifact_uid,
        tag=artifact_2_tag,
    )
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.read_artifact(db_session, artifact_1_key, artifact_2_tag)
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.read_artifact(db_session, artifact_2_key, artifact_1_tag)
    # just verifying it's not raising
    db.read_artifact(db_session, artifact_1_key, artifact_1_tag)
    db.read_artifact(db_session, artifact_2_key, artifact_2_tag)
    # check list
    artifacts = db.list_artifacts(db_session, tag=artifact_1_tag)
    assert len(artifacts) == 1
    artifacts = db.list_artifacts(db_session, tag=artifact_2_tag)
    assert len(artifacts) == 1


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_delete_artifacts_tag_filter(db: DBInterface, db_session: Session):
    artifact_1_key = "artifact_key_1"
    artifact_2_key = "artifact_key_2"
    artifact_1_uid = "artifact_uid_1"
    artifact_2_uid = "artifact_uid_2"
    artifact_1_body = _generate_artifact(artifact_1_key, uid=artifact_1_uid)
    artifact_2_body = _generate_artifact(artifact_2_key, uid=artifact_2_uid)
    artifact_1_tag = "artifact_tag_one"
    artifact_2_tag = "artifact_tag_two"

    db.store_artifact(
        db_session,
        artifact_1_key,
        artifact_1_body,
        artifact_1_uid,
        tag=artifact_1_tag,
    )
    db.store_artifact(
        db_session,
        artifact_2_key,
        artifact_2_body,
        artifact_2_uid,
        tag=artifact_2_tag,
    )
    db.del_artifacts(db_session, tag=artifact_1_tag)
    artifacts = db.list_artifacts(db_session, tag=artifact_1_tag)
    assert len(artifacts) == 0
    artifacts = db.list_artifacts(db_session, tag=artifact_2_tag)
    assert len(artifacts) == 1
    db.del_artifacts(db_session, tag=artifact_2_uid)
    artifacts = db.list_artifacts(db_session, tag=artifact_2_tag)
    assert len(artifacts) == 0


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_artifacts_exact_name_match(db: DBInterface, db_session: Session):
    artifact_1_key = "pre_artifact_key_suffix"
    artifact_2_key = "pre-artifact-key-suffix"
    artifact_1_uid = "artifact_uid_1"
    artifact_2_uid = "artifact_uid_2"
    artifact_1_body = _generate_artifact(artifact_1_key, uid=artifact_1_uid)
    artifact_2_body = _generate_artifact(artifact_2_key, uid=artifact_2_uid)

    # Store each twice - once with no iter, and once with an iter
    db.store_artifact(
        db_session,
        artifact_1_key,
        artifact_1_body,
        artifact_1_uid,
    )
    artifact_1_body["iter"] = 42
    db.store_artifact(
        db_session,
        artifact_1_key,
        artifact_1_body,
        artifact_1_uid,
        iter=42,
    )
    db.store_artifact(
        db_session,
        artifact_2_key,
        artifact_2_body,
        artifact_2_uid,
    )
    artifact_2_body["iter"] = 42
    db.store_artifact(
        db_session,
        artifact_2_key,
        artifact_2_body,
        artifact_2_uid,
        iter=42,
    )

    def _list_and_assert_count(key, count, iter=None):
        results = db.list_artifacts(db_session, name=key, iter=iter)
        assert len(results) == count
        return results

    # Ensure fuzzy query works, and we have everything we need
    _list_and_assert_count("~key", count=4)

    # Do an exact match with underscores in the name - must escape the _ do it doesn't do a like query
    list_results = _list_and_assert_count(artifact_1_key, count=2)
    for artifact in list_results:
        assert artifact["metadata"]["name"] == artifact_1_key

    _list_and_assert_count("%key%", count=0)
    # Verify we don't get artifacts whose name is "%-suffix" due to the like query used in the DB
    _list_and_assert_count("suffix", count=0)
    # This should also be filtered, since the prefix is "pre" which is 3 chars. There's a known caveat if
    # prefix is 1 or 2 chars long.
    _list_and_assert_count("artifact-key-suffix", count=0)

    _list_and_assert_count(artifact_1_key, iter=42, count=1)
    _list_and_assert_count("~key", iter=42, count=2)
    _list_and_assert_count("~key", iter=666, count=0)


def _generate_artifact_with_iterations(
    db, db_session, key, uid, num_iters, best_iter, kind
):
    for iter in range(num_iters):
        artifact_body = _generate_artifact(
            key, kind=kind.value if iter != 0 else "link", uid=uid
        )
        if iter == 0:
            artifact_body["spec"]["link_iteration"] = best_iter
        artifact_body["spec"]["iter"] = iter
        db.store_artifact(
            db_session,
            key,
            artifact_body,
            uid,
            iter=iter,
        )


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "db,db_session", [(dbs[0], dbs[0])], indirect=["db", "db_session"]
)
def test_list_artifacts_best_iter(db: DBInterface, db_session: Session):
    artifact_1_key = "artifact-1"
    artifact_1_uid = "uid-1"
    artifact_2_key = "artifact-2"
    artifact_2_uid = "uid-2"
    artifact_no_link_key = "single-artifact"
    artifact_no_link_uid = "uid-3"

    num_iters = 5
    best_iter_1 = 2
    best_iter_2 = 4
    _generate_artifact_with_iterations(
        db,
        db_session,
        artifact_1_key,
        artifact_1_uid,
        num_iters,
        best_iter_1,
        ArtifactCategories.model,
    )
    _generate_artifact_with_iterations(
        db,
        db_session,
        artifact_2_key,
        artifact_2_uid,
        num_iters,
        best_iter_2,
        ArtifactCategories.dataset,
    )

    # Add non-hyper-param artifact. Single object with iter 0, not pointing at anything
    artifact_body = _generate_artifact(artifact_no_link_key, artifact_no_link_uid)
    artifact_body["spec"]["iter"] = 0
    db.store_artifact(
        db_session, artifact_no_link_key, artifact_body, artifact_no_link_uid, iter=0
    )

    results = db.list_artifacts(db_session, name="~artifact")
    assert len(results) == num_iters * 2 + 1

    results = db.list_artifacts(db_session, name=artifact_1_key, best_iteration=True)
    assert len(results) == 1 and results[0]["spec"]["iter"] == best_iter_1

    expected_iters = {
        artifact_1_key: best_iter_1,
        artifact_2_key: best_iter_2,
        artifact_no_link_key: 0,
    }
    results = db.list_artifacts(db_session, name="~artifact", best_iteration=True)
    assert len(results) == 3
    for artifact in results:
        artifact_name = artifact["metadata"]["name"]
        assert (
            artifact_name in expected_iters
            and expected_iters[artifact_name] == artifact["spec"]["iter"]
        )

    results = db.list_artifacts(
        db_session, best_iteration=True, category=ArtifactCategories.model
    )
    assert len(results) == 1 and results[0]["spec"]["iter"] == best_iter_1

    # Should get only object-2 (which is of dataset type) and the link artifact
    results = db.list_artifacts(db_session, category=ArtifactCategories.dataset)
    assert len(results) == num_iters
    for artifact in results:
        assert artifact["metadata"]["name"] == artifact_2_key

    # Negative test - asking for both best_iter and iter
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        results = db.list_artifacts(
            db_session, name="~artifact", best_iteration=True, iter=0
        )


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "data_migration_db,db_session",
    [(dbs[0], dbs[0])],
    indirect=["data_migration_db", "db_session"],
)
def test_data_migration_fix_artifact_tags_duplications(
    data_migration_db: DBInterface,
    db_session: Session,
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

    data_migration_db.tag_artifacts = _buggy_tag_artifacts

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

    data_migration_db.store_artifact(
        db_session,
        artifact_1_key,
        artifact_1_body,
        artifact_1_uid,
    )
    data_migration_db.store_artifact(
        db_session,
        artifact_1_key,
        artifact_1_with_kind_body,
        artifact_1_with_kind_uid,
    )
    data_migration_db.store_artifact(
        db_session, artifact_2_key, artifact_2_body, artifact_2_uid, tag="not-latest"
    )
    data_migration_db.store_artifact(
        db_session,
        artifact_2_key,
        artifact_2_with_kind_body,
        artifact_2_with_kind_uid,
        tag="not-latest",
    )
    data_migration_db.store_artifact(
        db_session, artifact_3_key, artifact_3_with_kind_body, artifact_3_with_kind_uid
    )

    # Before the migration:
    # 1. read artifact would have failed when there's more than one tag record with the same key (happen when you
    # store twice)
    with pytest.raises(MultipleResultsFound):
        data_migration_db.read_artifact(db_session, artifact_1_key, tag="latest")
    with pytest.raises(MultipleResultsFound):
        data_migration_db.read_artifact(db_session, artifact_2_key, tag="not-latest")

    # 2. read artifact would have succeed when there's only one tag record with the same key (happen when you
    # stored only once)
    artifact = data_migration_db.read_artifact(db_session, artifact_3_key, tag="latest")
    assert artifact["metadata"]["uid"] == artifact_3_with_kind_uid

    # 3. list artifact without tag would have returned the latest (by update time) of each artifact key
    artifacts = data_migration_db.list_artifacts(db_session)
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
    artifacts = data_migration_db.list_artifacts(db_session, tag="latest")
    assert len(artifacts) == len(
        [artifact_1_uid, artifact_1_with_kind_uid, artifact_3_with_kind_uid]
    )

    # perform the migration
    mlrun.api.initial_data._fix_artifact_tags_duplications(
        data_migration_db, db_session
    )

    # After the migration:
    # 1. read artifact should succeed (fixed) and return the latest updated record that was tagged with the requested
    # tag
    artifact = data_migration_db.read_artifact(db_session, artifact_1_key, tag="latest")
    assert artifact["metadata"]["uid"] == artifact_1_with_kind_uid
    artifact = data_migration_db.read_artifact(
        db_session, artifact_2_key, tag="not-latest"
    )
    assert artifact["metadata"]["uid"] == artifact_2_with_kind_uid

    # 2. read artifact should (still) succeed when there's only one tag record with the same key (happen when you
    # stored only once)
    artifact = data_migration_db.read_artifact(db_session, artifact_3_key, tag="latest")
    assert artifact["metadata"]["uid"] == artifact_3_with_kind_uid

    # 3. list artifact without tag should (still) return the latest (by update time) of each artifact key
    artifacts = data_migration_db.list_artifacts(db_session)
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
    artifacts = data_migration_db.list_artifacts(db_session, tag="latest")
    assert (
        deepdiff.DeepDiff(
            [artifact["metadata"]["uid"] for artifact in artifacts],
            [artifact_1_with_kind_uid, artifact_3_with_kind_uid],
            ignore_order=True,
        )
        == {}
    )


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "data_migration_db,db_session",
    [(dbs[0], dbs[0])],
    indirect=["data_migration_db", "db_session"],
)
def test_data_migration_fix_legacy_datasets_large_previews(
    data_migration_db: DBInterface,
    db_session: Session,
):
    artifact_with_valid_preview_key = "artifact-with-valid-preview-key"
    artifact_with_valid_preview_uid = "artifact-with-valid-preview-uid"
    artifact_with_valid_preview = mlrun.artifacts.dataset.LegacyDatasetArtifact(
        artifact_with_valid_preview_key,
        df=pandas.DataFrame(
            [{"A": 10, "B": 100}, {"A": 11, "B": 110}, {"A": 12, "B": 120}]
        ),
    )
    data_migration_db._store_artifact(
        db_session,
        artifact_with_valid_preview_key,
        artifact_with_valid_preview.to_dict(),
        artifact_with_valid_preview_uid,
    )

    artifact_with_invalid_preview_key = "artifact-with-invalid-preview-key"
    artifact_with_invalid_preview_uid = "artifact-with-invalid-preview-uid"
    artifact_with_invalid_preview = mlrun.artifacts.dataset.LegacyDatasetArtifact(
        artifact_with_invalid_preview_key,
        df=pandas.DataFrame(
            numpy.random.randint(
                0, 10, size=(10, mlrun.artifacts.dataset.max_preview_columns * 3)
            )
        ),
        ignore_preview_limits=True,
    )
    data_migration_db._store_artifact(
        db_session,
        artifact_with_invalid_preview_key,
        artifact_with_invalid_preview.to_dict(),
        artifact_with_invalid_preview_uid,
    )

    # perform the migration
    mlrun.api.initial_data._fix_datasets_large_previews(data_migration_db, db_session)

    artifact_with_valid_preview_after_migration = data_migration_db.read_artifact(
        db_session, artifact_with_valid_preview_key, artifact_with_valid_preview_uid
    )
    assert (
        deepdiff.DeepDiff(
            artifact_with_valid_preview_after_migration,
            artifact_with_valid_preview.to_dict(),
            ignore_order=True,
            exclude_paths=["root['updated']", "root['tag']", "root['db_key']"],
        )
        == {}
    )

    artifact_with_invalid_preview_after_migration = data_migration_db.read_artifact(
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
                "root['tag']",
                "root['db_key']",
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


# running only on sqldb cause filedb is not really a thing anymore, will be removed soon
@pytest.mark.parametrize(
    "data_migration_db,db_session",
    [(dbs[0], dbs[0])],
    indirect=["data_migration_db", "db_session"],
)
def test_data_migration_fix_datasets_large_previews(
    data_migration_db: DBInterface,
    db_session: Session,
):
    artifact_with_valid_preview_key = "artifact-with-valid-preview-key"
    artifact_with_valid_preview_uid = "artifact-with-valid-preview-uid"
    artifact_with_valid_preview = mlrun.artifacts.dataset.DatasetArtifact(
        artifact_with_valid_preview_key,
        df=pandas.DataFrame(
            [{"A": 10, "B": 100}, {"A": 11, "B": 110}, {"A": 12, "B": 120}]
        ),
    )
    data_migration_db._store_artifact(
        db_session,
        artifact_with_valid_preview_key,
        artifact_with_valid_preview.to_dict(),
        artifact_with_valid_preview_uid,
    )

    artifact_with_invalid_preview_key = "artifact-with-invalid-preview-key"
    artifact_with_invalid_preview_uid = "artifact-with-invalid-preview-uid"
    artifact_with_invalid_preview = mlrun.artifacts.dataset.DatasetArtifact(
        artifact_with_invalid_preview_key,
        df=pandas.DataFrame(
            numpy.random.randint(
                0, 10, size=(10, mlrun.artifacts.dataset.max_preview_columns * 3)
            )
        ),
        ignore_preview_limits=True,
    )
    data_migration_db._store_artifact(
        db_session,
        artifact_with_invalid_preview_key,
        artifact_with_invalid_preview.to_dict(),
        artifact_with_invalid_preview_uid,
    )

    # perform the migration
    mlrun.api.initial_data._fix_datasets_large_previews(data_migration_db, db_session)

    artifact_with_valid_preview_after_migration = data_migration_db.read_artifact(
        db_session, artifact_with_valid_preview_key, artifact_with_valid_preview_uid
    )
    assert (
        deepdiff.DeepDiff(
            artifact_with_valid_preview_after_migration,
            artifact_with_valid_preview.to_dict(),
            ignore_order=True,
            exclude_paths=[
                "root['metadata']['updated']",
                "root['metadata']['tag']",
                "root['spec']['db_key']",
            ],
        )
        == {}
    )

    artifact_with_invalid_preview_after_migration = data_migration_db.read_artifact(
        db_session, artifact_with_invalid_preview_key, artifact_with_invalid_preview_uid
    )
    assert (
        deepdiff.DeepDiff(
            artifact_with_invalid_preview_after_migration,
            artifact_with_invalid_preview.to_dict(),
            ignore_order=True,
            exclude_paths=[
                "root['metadata']['updated']",
                "root['spec']['header']",
                "root['status']['stats']",
                "root['spec']['schema']",
                "root['status']['preview']",
                "root['metadata']['tag']",
                "root['spec']['db_key']",
            ],
        )
        == {}
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["spec"]["header"])
        == mlrun.artifacts.dataset.max_preview_columns
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["status"]["stats"])
        == mlrun.artifacts.dataset.max_preview_columns - 1
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["status"]["preview"][0])
        == mlrun.artifacts.dataset.max_preview_columns
    )
    assert (
        len(artifact_with_invalid_preview_after_migration["spec"]["schema"]["fields"])
        == mlrun.artifacts.dataset.max_preview_columns + 1
    )


def _generate_artifact(name, uid=None, kind=None):
    artifact = {
        "metadata": {"name": name},
        "spec": {"src_path": "/some/path"},
        "kind": kind,
        "status": {"bla": "blabla"},
    }
    if kind:
        artifact["kind"] = kind
    if uid:
        artifact["metadata"]["uid"] = uid

    return artifact
