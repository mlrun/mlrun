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

from datetime import datetime, timedelta, timezone
from tempfile import mkdtemp

import pytest

from mlrun.db import FileRunDB


@pytest.fixture
def db():
    path = mkdtemp(prefix="mlrun-test")
    db = FileRunDB(dirpath=path)
    db.connect()
    return db


def test_save_get_function(db: FileRunDB):
    func, name, proj = {"x": 1, "y": 2}, "f1", "p2"
    db.store_function(func, name, proj)
    db_func = db.get_function(name, proj)

    # db methods enriches metadata
    del db_func["metadata"]
    del func["metadata"]
    assert db_func == func, "wrong func"


def test_list_functions(db: FileRunDB):
    proj = "p4"
    count = 5
    for i in range(count):
        name = f"func{i}"
        func = {"fid": i}
        db.store_function(func, name, proj)
    db.store_function({}, "f2", "p7")

    out = db.list_functions("", proj)
    assert len(out) == count, "bad list"


def test_schedules(db: FileRunDB):
    count = 7
    for i in range(count):
        data = {"i": i}
        db.store_schedule(data)

    scheds = list(db.list_schedules())
    assert count == len(scheds), "wrong number of schedules"
    assert set(range(count)) == set(s["i"] for s in scheds), "bad scheds"


def test_list_artifact_date(db: FileRunDB):
    print("dirpath: ", db.dirpath)
    t1 = datetime(2020, 2, 16, tzinfo=timezone.utc)
    t2 = t1 - timedelta(days=7)
    t3 = t2 - timedelta(days=7)
    prj = "p7"

    db.store_artifact("k1", {"updated": t1.isoformat()}, "u1", project=prj)
    db.store_artifact("k2", {"updated": t2.isoformat()}, "u2", project=prj)
    db.store_artifact("k3", {"updated": t3.isoformat()}, "u3", project=prj)

    # FIXME: We get double what we expect since latest is an alias
    arts = db.list_artifacts(project=prj, since=t3, tag="*")
    assert 6 == len(arts), "since t3"

    arts = db.list_artifacts(project=prj, since=t2, tag="*")
    assert 4 == len(arts), "since t2"

    arts = db.list_artifacts(project=prj, since=t1 + timedelta(days=1), tag="*")
    assert not arts, "since t1+"

    arts = db.list_artifacts(project=prj, until=t2, tag="*")
    assert 4 == len(arts), "until t2"

    arts = db.list_artifacts(project=prj, since=t2, until=t2, tag="*")
    assert 2 == len(arts), "since/until t2"
