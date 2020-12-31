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

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta

import deepdiff
import pytest
from sqlalchemy.orm import Session

import mlrun.api.schemas
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.db.sqldb.models import _tagged
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

    tags = db.list_artifact_tags(db_session, "p1")
    assert {"t1", "t2"} == set(tags), "bad tags"


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


def test_list_projects(db: SQLDB, db_session: Session):
    for i in range(10):
        run = new_run("s1", {"l1": "v1", "l2": "v2"}, x=1)
        db.store_run(db_session, run, "u7", project=f"prj{i % 3}", iter=i)

    projects_output = db.list_projects(db_session)

    assert {"prj0", "prj1", "prj2"} == {
        project.metadata.name for project in projects_output.projects
    }


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
    assert 2 == len(arts), "count"
    assert art2["a"] == arts[1]["a"], "bad artifact"

    k2, u3, art3 = "k2", "u3", {"a": 99}
    db.store_artifact(db_session, k2, art3, u3, project=prj)
    arts = db.list_artifacts(db_session, project=prj, tag="latest")
    assert 3 == len(arts), "number"
    assert {1, 17, 99} == set(art["a"] for art in arts), "latest"


@pytest.mark.parametrize("cls", _tagged)
def test_tags(db: SQLDB, db_session: Session, cls):
    p1, n1 = "prj1", "name1"
    obj1, obj2, obj3 = cls(), cls(), cls()
    db_session.add(obj1)
    db_session.add(obj2)
    db_session.add(obj3)
    db_session.commit()

    db.tag_objects(db_session, [obj1, obj2], p1, n1)
    objs = db.find_tagged(db_session, p1, n1)
    assert {obj1, obj2} == set(objs), "find tags"

    db.del_tag(db_session, p1, n1)
    objs = db.find_tagged(db_session, p1, n1)
    assert [] == objs, "find tags after del"


def _tag_objs(db: SQLDB, db_session: Session, count, project, tags):
    by_tag = defaultdict(list)
    for i in range(count):
        cls = _tagged[i % len(_tagged)]
        obj = cls()
        by_tag[tags[i % len(tags)]].append(obj)
        db_session.add(obj)
    db_session.commit()
    for tag, objs in by_tag.items():
        db.tag_objects(db_session, objs, project, tag)


def test_list_tags(db: SQLDB, db_session: Session):
    p1, tags1 = "prj1", ["a", "b", "c"]
    _tag_objs(db, db_session, 17, p1, tags1)
    p2, tags2 = "prj2", ["b", "c", "d", "e"]
    _tag_objs(db, db_session, 11, p2, tags2)

    tags = db.list_tags(db_session, p1)
    assert set(tags) == set(tags1), "tags"


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
            project.dict(), project_output.dict(exclude={"id"}), ignore_order=True,
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
        db_session, format_=mlrun.api.schemas.Format.name_only
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
