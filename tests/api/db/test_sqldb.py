# Copyright 2023 Iguazio
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
"""SQLDB specific tests, common tests should be in test_dbs.py"""

import copy
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest import mock

import deepdiff
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import mlrun.artifacts
import mlrun.common.formatters
import mlrun.common.schemas
import server.api.db.sqldb.models
from mlrun.lists import ArtifactList
from server.api.db.sqldb.db import SQLDB
from server.api.db.sqldb.models import ArtifactV2
from tests.conftest import new_run


@contextmanager
def patch(obj, **kw):
    old = {}
    for k, v in kw.items():
        old[k] = getattr(obj, k)
        setattr(obj, k, v)
    try:
        yield obj
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def test_list_artifact_tags(db: SQLDB, db_session: Session):
    db.store_artifact(db_session, "k1", {}, producer_id="1", tag="t1", project="p1")
    db.store_artifact(db_session, "k1", {}, producer_id="2", tag="t2", project="p1")
    db.store_artifact(db_session, "k1", {}, producer_id="2", tag="t2", project="p2")
    db.store_artifact(
        db_session, "k2", {"kind": "model"}, producer_id="3", tag="t3", project="p1"
    )
    db.store_artifact(
        db_session, "k3", {"kind": "dataset"}, producer_id="4", tag="t4", project="p2"
    )

    tags = db.list_artifact_tags(db_session, "p1")
    expected_tags = [
        "t1",
        "latest",
        "t2",
        "t3",
    ]
    assert deepdiff.DeepDiff(tags, expected_tags, ignore_order=True) == {}

    # filter by category
    model_tags = db.list_artifact_tags(
        db_session, "p1", mlrun.common.schemas.ArtifactCategories.model
    )
    expected_tags = ["t3", "latest"]
    assert deepdiff.DeepDiff(expected_tags, model_tags, ignore_order=True) == {}

    model_tags = db.list_artifact_tags(
        db_session, "p2", mlrun.common.schemas.ArtifactCategories.dataset
    )
    expected_tags = ["t4", "latest"]
    assert deepdiff.DeepDiff(expected_tags, model_tags, ignore_order=True) == {}


def test_list_artifact_date(db: SQLDB, db_session: Session):
    t1 = datetime(2020, 2, 16)
    t2 = t1 - timedelta(days=7)
    t3 = t2 - timedelta(days=7)
    project = "p7"

    # create artifacts in the db directly to avoid the store_artifact function which sets the updated field
    artifacts_to_create = []
    for key, updated, producer_id in [
        ("k1", t1, "p1"),
        ("k2", t2, "p2"),
        ("k3", t3, "p3"),
    ]:
        artifact_struct = mlrun.artifacts.Artifact(
            metadata=mlrun.artifacts.ArtifactMetadata(
                key=key, project=project, tree=producer_id
            ),
            spec=mlrun.artifacts.ArtifactSpec(),
        )
        db_artifact = ArtifactV2(
            project=project, key=key, updated=updated, producer_id=producer_id
        )
        db_artifact.full_object = artifact_struct.to_dict()
        artifacts_to_create.append(db_artifact)

    db._upsert(db_session, artifacts_to_create)

    arts = db.list_artifacts(db_session, project=project, since=t3, tag="*")
    assert 3 == len(arts), "since t3"

    arts = db.list_artifacts(db_session, project=project, since=t2, tag="*")
    assert 2 == len(arts), "since t2"

    arts = db.list_artifacts(
        db_session, project=project, since=t1 + timedelta(days=1), tag="*"
    )
    assert not arts, "since t1+"

    arts = db.list_artifacts(db_session, project=project, until=t2, tag="*")
    assert 2 == len(arts), "until t2"

    arts = db.list_artifacts(db_session, project=project, since=t2, until=t2, tag="*")
    assert 1 == len(arts), "since/until t2"


def test_run_iter0(db: SQLDB, db_session: Session):
    uid, prj = "uid39", "lemon"
    run = new_run("s1", {"l1": "v1", "l2": "v2"}, x=1)
    for i in range(7):
        db.store_run(db_session, run, uid, prj, i)
    db._get_run(db_session, uid, prj, 0)  # See issue 140


def test_artifacts_latest(db: SQLDB, db_session: Session):
    k1, t1, art1 = "k1", "t1", {"a": 1}
    prj = "p38"
    db.store_artifact(db_session, k1, art1, producer_id=t1, project=prj)

    arts = db.list_artifacts(db_session, project=prj, tag="latest")
    assert art1["a"] == arts[0]["a"], "bad artifact"

    t2, art2 = "t2", {"a": 17}
    db.store_artifact(db_session, k1, art2, producer_id=t2, project=prj)
    arts = db.list_artifacts(db_session, project=prj, tag="latest")
    assert 1 == len(arts), "count"
    assert art2["a"] == arts[0]["a"], "bad artifact"

    k2, t3, art3 = "k2", "t3", {"a": 99}
    db.store_artifact(db_session, k2, art3, producer_id=t3, project=prj)
    arts = db.list_artifacts(db_session, project=prj, tag="latest")
    assert 2 == len(arts), "number"
    assert {17, 99} == set(art["a"] for art in arts), "latest"


def test_read_and_list_artifacts_with_tags(db: SQLDB, db_session: Session):
    k1, t1, art1 = "k1", "t1", {"a": 1, "b": "blubla"}
    t2, art2 = "t2", {"a": 2, "b": "blublu"}
    prj = "p38"
    db.store_artifact(
        db_session, k1, art1, producer_id=t1, iter=1, project=prj, tag="tag1"
    )
    db.store_artifact(
        db_session, k1, art2, producer_id=t2, iter=2, project=prj, tag="tag2"
    )

    result = db.read_artifact(db_session, k1, "tag1", iter=1, project=prj)
    assert result["metadata"]["tag"] == "tag1"
    result = db.read_artifact(db_session, k1, "tag2", iter=2, project=prj)
    assert result["metadata"]["tag"] == "tag2"
    result = db.read_artifact(
        db_session,
        k1,
        iter=1,
        project=prj,
        tag=mlrun.common.schemas.artifact.ArtifactTagsTypes.untagged,
    )
    # When doing get without a tag, the returned object must not contain a tag.
    assert "tag" not in result["metadata"]

    result = db.list_artifacts(db_session, k1, project=prj, tag="*")
    assert len(result) == 3
    for artifact in result:
        assert (
            (artifact["a"] == 1 and artifact["metadata"]["tag"] == "tag1")
            or (artifact["a"] == 2 and artifact["metadata"]["tag"] == "tag2")
            or (artifact["a"] in (1, 2) and artifact["metadata"]["tag"] == "latest")
        )

    # To be used later, after adding tags
    full_results = result

    result = db.list_artifacts(db_session, k1, tag="tag1", project=prj)
    assert (
        len(result) == 1
        and result[0]["metadata"]["tag"] == "tag1"
        and result[0]["a"] == 1
    )
    result = db.list_artifacts(db_session, k1, tag="tag2", project=prj)
    assert (
        len(result) == 1
        and result[0]["metadata"]["tag"] == "tag2"
        and result[0]["a"] == 2
    )

    # Add another tag to all objects (there are 2 at this point)
    new_tag = "new-tag"
    expected_results = ArtifactList()
    for artifact in full_results:
        expected_results.append(artifact)
        if artifact["metadata"]["tag"] == "latest":
            # We don't want to add a new tag to the "latest" object (it's the same object as the one with tag "tag2")
            continue
        artifact_with_new_tag = copy.deepcopy(artifact)
        artifact_with_new_tag["metadata"]["tag"] = new_tag
        expected_results.append(artifact_with_new_tag)

    artifacts = db_session.query(ArtifactV2).all()
    db.tag_objects_v2(
        db_session, artifacts, prj, name=new_tag, obj_name_attribute="key"
    )
    result = db.list_artifacts(db_session, k1, prj, tag="*")
    assert deepdiff.DeepDiff(result, expected_results, ignore_order=True) == {}

    # Add another tag to the art1
    db.store_artifact(
        db_session, k1, art1, producer_id=t1, iter=1, project=prj, tag="tag3"
    )
    # this makes it the latest object of this key, so we need to remove the artifact
    # with tag "latest" from the expected results
    expected_results = ArtifactList(
        [
            artifact
            for artifact in expected_results
            if artifact["metadata"]["tag"] != "latest"
        ]
    )

    result = db.read_artifact(db_session, k1, "tag3", iter=1, project=prj)
    assert result["metadata"]["tag"] == "tag3"
    expected_results.append(copy.deepcopy(result))

    # add it again but with the "latest" tag
    result["metadata"]["tag"] = "latest"
    expected_results.append(result)

    result = db.list_artifacts(db_session, k1, prj, tag="*")
    # We want to ignore the "updated" field, since it changes as we store a new tag.
    exclude_regex = r"root\[\d+\]\['updated'\]"
    assert (
        deepdiff.DeepDiff(
            result,
            expected_results,
            ignore_order=True,
            exclude_regex_paths=exclude_regex,
        )
        == {}
    )


def test_read_untagged_artifact(db: SQLDB, db_session: Session):
    project = "dummy-project"
    key = "dummy-key"

    for index in range(1, 5):
        db.store_artifact(
            db_session,
            key,
            artifact={"a": index},
            iter=index,
            project=project,
            producer_id=f"dummy-id-{index}",
        )

    artifacts = db.list_artifacts(db_session, project=project)
    assert len(artifacts) == 4

    non_latest_artifacts = [
        artifact
        for artifact in artifacts
        if artifact["metadata"].get("tag") != "latest"
    ]

    assert (
        len(non_latest_artifacts) == 3
    ), "There should be exactly 3 non-latest artifact"

    #  Retrieve the oldest artifact (the first one) by using the 'untagged' tag
    retrieved_artifact = db.read_artifact(
        db_session,
        key,
        iter=1,
        project=project,
        producer_id="dummy-id-1",
        tag=mlrun.common.schemas.artifact.ArtifactTagsTypes.untagged,
    )

    # Verify that we retrieved the correct artifact by checking its body
    assert retrieved_artifact["a"] == 1, "The retrieved artifact should have the body {'a': 1}"


def test_projects_crud(db: SQLDB, db_session: Session):
    project = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name="p1"),
        spec=mlrun.common.schemas.ProjectSpec(
            description="banana", other_field="value"
        ),
        status=mlrun.common.schemas.ObjectStatus(state="active"),
    )
    db.create_project(db_session, project)
    project_output = db.get_project(db_session, name=project.metadata.name)
    assert (
        deepdiff.DeepDiff(
            project.dict(),
            project_output.dict(exclude={"id"}),
            ignore_order=True,
        )
        == {}
    )

    project_patch = {"spec": {"description": "lemon"}}
    db.patch_project(db_session, project.metadata.name, project_patch)
    project_output = db.get_project(db_session, name=project.metadata.name)
    assert project_output.spec.description == project_patch["spec"]["description"]

    project_2 = mlrun.common.schemas.Project(
        metadata=mlrun.common.schemas.ProjectMetadata(name="p2"),
    )
    db.create_project(db_session, project_2)
    projects_output = db.list_projects(
        db_session, format_=mlrun.common.formatters.ProjectFormat.name_only
    )
    assert [project.metadata.name, project_2.metadata.name] == projects_output.projects


@pytest.mark.parametrize(
    "error_message, expected_exception",
    [
        # exhausted retries
        ("database is locked", Exception),
        # conflicts
        (
            "(sqlite3.IntegrityError) UNIQUE constraint failed",
            mlrun.errors.MLRunConflictError,
        ),
        ("(pymysql.err.IntegrityError) (1062", mlrun.errors.MLRunConflictError),
        ("(pymysql.err.IntegrityError) (1586", mlrun.errors.MLRunConflictError),
        # other errors
        ("some other exception", mlrun.errors.MLRunRuntimeError),
    ],
)
def test_commit_failures(db: SQLDB, error_message: str, expected_exception: Exception):
    # create some fake objects to commit
    objects = [
        server.api.db.sqldb.models.Run(project="p1", uid="u1", name="run-1"),
        server.api.db.sqldb.models.Feature(feature_set_id="fs-1", name="feat-1"),
        server.api.db.sqldb.models.Function(project="p3", name="func-1"),
    ]

    session = mock.MagicMock()
    session.commit = mock.MagicMock(side_effect=SQLAlchemyError(error_message))

    with pytest.raises(expected_exception):
        db._commit(session, objects)


# def test_function_latest(db: SQLDB, db_session: Session):
#     fn1, t1 = {'x': 1}, 'u83'
#     fn2, t2 = {'x': 2}, 'u23'
#     prj, name = 'p388', 'n3023'
#     db.store_function(db_session, fn1, name, prj, t1)
#     db.store_function(db_session, fn2, name, prj, t2)
#
#     fn = db.get_function(db_session, name, prj, 'latest')
#     assert fn2 == fn, 'latest'
