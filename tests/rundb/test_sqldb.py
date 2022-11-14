# Copyright 2019 Iguazio
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

from contextlib import contextmanager
from datetime import datetime, timedelta

import deepdiff
from sqlalchemy.orm import Session

import mlrun.api.schemas
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.db.sqldb.models import Artifact
from mlrun.lists import ArtifactList
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
    db.store_artifact(db_session, "k1", {}, "1", tag="t1", project="p1")
    db.store_artifact(db_session, "k1", {}, "2", tag="t2", project="p1")
    db.store_artifact(db_session, "k1", {}, "2", tag="t2", project="p2")
    db.store_artifact(db_session, "k2", {"kind": "model"}, "3", tag="t3", project="p1")
    db.store_artifact(
        db_session, "k3", {"kind": "dataset"}, "4", tag="t4", project="p2"
    )

    tags = db.list_artifact_tags(db_session, "p1")
    assert [("p1", "k1", "t1"), ("p1", "k1", "t2"), ("p1", "k2", "t3")] == tags

    # filter by category
    model_tags = db.list_artifact_tags(
        db_session, "p1", mlrun.api.schemas.ArtifactCategories.model
    )
    assert [("p1", "k2", "t3")] == model_tags

    model_tags = db.list_artifact_tags(
        db_session, "p2", mlrun.api.schemas.ArtifactCategories.dataset
    )
    assert [("p2", "k3", "t4")] == model_tags


def test_list_artifact_date(db: SQLDB, db_session: Session):
    t1 = datetime(2020, 2, 16)
    t2 = t1 - timedelta(days=7)
    t3 = t2 - timedelta(days=7)
    prj = "p7"

    db.store_artifact(db_session, "k1", {"updated": t1}, "u1", project=prj)
    db.store_artifact(db_session, "k2", {"updated": t2}, "u2", project=prj)
    db.store_artifact(db_session, "k3", {"updated": t3}, "u3", project=prj)

    arts = db.list_artifacts(db_session, project=prj, since=t3, tag="*")
    assert 3 == len(arts), "since t3"

    arts = db.list_artifacts(db_session, project=prj, since=t2, tag="*")
    assert 2 == len(arts), "since t2"

    arts = db.list_artifacts(
        db_session, project=prj, since=t1 + timedelta(days=1), tag="*"
    )
    assert not arts, "since t1+"

    arts = db.list_artifacts(db_session, project=prj, until=t2, tag="*")
    assert 2 == len(arts), "until t2"

    arts = db.list_artifacts(db_session, project=prj, since=t2, until=t2, tag="*")
    assert 1 == len(arts), "since/until t2"


def test_run_iter0(db: SQLDB, db_session: Session):
    uid, prj = "uid39", "lemon"
    run = new_run("s1", {"l1": "v1", "l2": "v2"}, x=1)
    for i in range(7):
        db.store_run(db_session, run, uid, prj, i)
    db._get_run(db_session, uid, prj, 0)  # See issue 140


def test_artifacts_latest(db: SQLDB, db_session: Session):
    k1, u1, art1 = "k1", "u1", {"a": 1}
    prj = "p38"
    db.store_artifact(db_session, k1, art1, u1, project=prj)

    arts = db.list_artifacts(db_session, project=prj, tag="latest")
    assert art1["a"] == arts[0]["a"], "bad artifact"

    u2, art2 = "u2", {"a": 17}
    db.store_artifact(db_session, k1, art2, u2, project=prj)
    arts = db.list_artifacts(db_session, project=prj, tag="latest")
    assert 1 == len(arts), "count"
    assert art2["a"] == arts[0]["a"], "bad artifact"

    k2, u3, art3 = "k2", "u3", {"a": 99}
    db.store_artifact(db_session, k2, art3, u3, project=prj)
    arts = db.list_artifacts(db_session, project=prj, tag="latest")
    assert 2 == len(arts), "number"
    assert {17, 99} == set(art["a"] for art in arts), "latest"


def test_read_and_list_artifacts_with_tags(db: SQLDB, db_session: Session):
    k1, u1, art1 = "k1", "u1", {"a": 1, "b": "blubla"}
    u2, art2 = "u2", {"a": 2, "b": "blublu"}
    prj = "p38"
    db.store_artifact(db_session, k1, art1, u1, iter=1, project=prj, tag="tag1")
    db.store_artifact(db_session, k1, art2, u2, iter=2, project=prj, tag="tag2")

    result = db.read_artifact(db_session, k1, "tag1", iter=1, project=prj)
    assert result["tag"] == "tag1"
    result = db.read_artifact(db_session, k1, "tag2", iter=2, project=prj)
    assert result["tag"] == "tag2"
    result = db.read_artifact(db_session, k1, iter=1, project=prj)
    # When doing get without a tag, the returned object must not contain a tag.
    assert "tag" not in result
    # read_artifact supports a case where the tag is actually the uid.
    result = db.read_artifact(db_session, k1, tag="u2", iter=2, project=prj)
    assert "tag" not in result

    result = db.read_artifact(db_session, k1, "tag2", iter=2, project=prj)
    assert result["tag"] == "tag2"

    result = db.list_artifacts(db_session, k1, project=prj, tag="*")
    # TODO: this actually expected to be 3, but this will be fixed in another PR once we restructure the Artifacts table
    #   and change the way tag_artifacts works ( currently it is unable to untag tag from artifacts which were generated
    #   in hyper params runs)
    assert len(result) == 4
    for artifact in result:
        assert (
            (artifact["a"] == 1 and artifact["tag"] == "tag1")
            or (artifact["a"] == 2 and artifact["tag"] == "tag2")
            or (artifact["a"] in (1, 2) and artifact["tag"] == "latest")
        )

    # To be used later, after adding tags
    full_results = result

    result = db.list_artifacts(db_session, k1, tag="tag1", project=prj)
    assert len(result) == 1 and result[0]["tag"] == "tag1" and result[0]["a"] == 1
    result = db.list_artifacts(db_session, k1, tag="tag2", project=prj)
    assert len(result) == 1 and result[0]["tag"] == "tag2" and result[0]["a"] == 2

    # Add another tag to all objects (there are 2 at this point)
    expected_results = ArtifactList()
    for artifact in full_results:
        expected_results.append(artifact)
        artifact_with_new_tag = artifact.copy()
        artifact_with_new_tag["tag"] = "new_tag"
        expected_results.append(artifact_with_new_tag)

    artifacts = db_session.query(Artifact).all()
    db.tag_artifacts(db_session, artifacts, prj, "new_tag")
    result = db.list_artifacts(db_session, k1, prj, tag="*")
    assert deepdiff.DeepDiff(result, expected_results, ignore_order=True) == {}

    db.store_artifact(db_session, k1, art1, u1, iter=1, project=prj, tag="tag3")
    result = db.read_artifact(db_session, k1, "tag3", iter=1, project=prj)
    assert result["tag"] == "tag3"
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


def test_projects_crud(db: SQLDB, db_session: Session):
    project = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name="p1"),
        spec=mlrun.api.schemas.ProjectSpec(description="banana", other_field="value"),
        status=mlrun.api.schemas.ObjectStatus(state="active"),
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

    project_2 = mlrun.api.schemas.Project(
        metadata=mlrun.api.schemas.ProjectMetadata(name="p2"),
    )
    db.create_project(db_session, project_2)
    projects_output = db.list_projects(
        db_session, format_=mlrun.api.schemas.ProjectsFormat.name_only
    )
    assert [project.metadata.name, project_2.metadata.name] == projects_output.projects


# def test_function_latest(db: SQLDB, db_session: Session):
#     fn1, t1 = {'x': 1}, 'u83'
#     fn2, t2 = {'x': 2}, 'u23'
#     prj, name = 'p388', 'n3023'
#     db.store_function(db_session, fn1, name, prj, t1)
#     db.store_function(db_session, fn2, name, prj, t2)
#
#     fn = db.get_function(db_session, name, prj, 'latest')
#     assert fn2 == fn, 'latest'
