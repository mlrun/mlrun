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

from tempfile import mkdtemp

import pytest

import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.errors
from mlrun.api.db.sqldb.session import _init_engine, create_session
from mlrun.api.initial_data import init_data
from mlrun.api.utils.singletons.db import initialize_db
from mlrun.config import config
from mlrun.db import SQLDB, FileRunDB, sqldb
from mlrun.db.base import RunDBInterface
from tests.conftest import new_run, run_now

dbs = [
    "sql",
    "file",
    # TODO: 'httpdb',
]


@pytest.fixture(params=dbs)
def db(request):
    path = mkdtemp()
    print(f"db fixture: path={path!r}")
    if request.param == "sql":
        db_file = f"{path}/mlrun.db"
        dsn = f"sqlite:///{db_file}?check_same_thread=false"
        config.httpdb.dsn = dsn
        _init_engine(dsn)
        init_data()
        initialize_db()
        db_session = create_session()
        db = SQLDB(dsn, session=db_session)
    elif request.param == "file":
        db = FileRunDB(path)
    else:
        assert False, f"unknown db type - {request.param}"

    db.connect()
    if request.param == "sql":
        mlrun.api.utils.singletons.db.initialize_db(db.db)
        mlrun.api.utils.singletons.project_member.initialize_project_member()
    return db


def test_save_get_function(db: RunDBInterface):
    func, name, proj, tag = {"data": {"x": 1, "y": 2}}, "f1", "p2", "t3u"
    db.store_function(func, name, proj, tag)
    db_func = db.get_function(name, proj, tag)

    # db methods enriches metadata
    assert func["data"] == db_func["data"]


def new_func(labels, **kw):
    obj = {
        "metadata": {"labels": labels},
    }
    obj.update(kw)
    return obj


def test_list_functions(db: RunDBInterface):
    name = "fn"
    fn1 = new_func({"l1": "v1", "l2": "v2"}, x=1)
    db.store_function(fn1, name)
    fn2 = new_func({"l2": "v2", "l3": "v3"}, x=2)
    db.store_function(fn2, name, tag="t1")
    fn3 = new_func({"l3": "v3"}, x=3)
    db.store_function(fn3, name, tag="t2")

    funcs = db.list_functions(name, labels={"l2": "v2"})
    assert 2 == len(funcs), "num of funcs"
    assert {1, 2} == {fn["x"] for fn in funcs}, "xs"


def test_runs(db: RunDBInterface):
    run1 = new_run("s1", {"l1": "v1", "l2": "v2"}, x=1)
    db.store_run(run1, "uid1")
    run2 = new_run("s1", {"l2": "v2", "l3": "v3"}, x=2)
    db.store_run(run2, "uid2")
    run3 = new_run("s2", {"l3": "v3"}, x=2)
    uid3 = "uid3"
    db.store_run(run3, uid3)
    db.store_run(run3, uid3)  # should not raise

    updates = {
        "status": {"start_time": run_now(), "state": "s2"},
    }
    db.update_run(updates, uid3)

    runs = db.list_runs(labels={"l2": "v2"})
    assert 2 == len(runs), "labels length"
    assert {1, 2} == {r["x"] for r in runs}, "xs labels"

    runs = db.list_runs(state=["s1", "s2"])
    assert 3 == len(runs), "state length"

    runs = db.list_runs(state="s2")
    assert 1 == len(runs), "state length"
    run3["status"] = updates["status"]
    assert run3 == runs[0], "state run"

    db.del_run(uid3)
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        db.read_run(uid3)

    label = "l1"
    db.del_runs(labels=[label])
    for run in db.list_runs():
        assert label not in run["metadata"]["labels"], "del_runs"


def test_update_run(db: sqldb.SQLDB):
    uid = "uid83"
    run = new_run("s1", {"l1": "v1", "l2": "v2"}, x=1)
    db.store_run(run, uid)
    val = 13
    db.update_run({"x": val}, uid)
    r = db.read_run(uid)
    assert val == r["x"], "bad update"


def test_artifacts(db: RunDBInterface):
    k1, u1, art1 = "k1", "u1", {"a": 1}
    db.store_artifact(k1, art1, u1)
    art = db.read_artifact(k1, u1)
    assert art1["a"] == art["a"], "get artifact"
    art = db.read_artifact(k1)
    assert art1["a"] == art["a"], "get latest artifact"

    prj = "p1"
    k2, u2, art2 = "k2", "u2", {"a": 2}
    db.store_artifact(k2, art2, u2, project=prj)
    k3, u3, art3 = "k3", "u3", {"a": 3}
    db.store_artifact(k3, art3, u3, project=prj)

    arts = db.list_artifacts(project=prj, tag="*")
    expected = 2 if isinstance(db, SQLDB) else 4  # FIXME
    assert expected == len(arts), "list artifacts length"
    assert {2, 3} == {a["a"] for a in arts}, "list artifact a"

    db.del_artifact(key=k1)
    with pytest.raises((sqldb.RunDBError, mlrun.errors.MLRunNotFoundError)):
        db.read_artifact(k1)


def test_list_runs(db: RunDBInterface):
    if isinstance(db, FileRunDB):
        pytest.skip("FIXME")
    uid = "u183"
    run = new_run("s1", {"l1": "v1", "l2": "v2"}, uid, x=1)
    count = 5
    for iter in range(count):
        db.store_run(run, uid, iter=iter)

    runs = list(db.list_runs(uid=uid))
    assert 1 == len(runs), "iter=False"

    runs = list(db.list_runs(uid=uid, iter=True))
    assert 5 == len(runs), "iter=True"
