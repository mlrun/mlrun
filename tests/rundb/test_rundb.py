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

import pytest

import mlrun
from tests.conftest import results, rundb_path


def get_db():
    return mlrun.get_run_db(rundb_path)


#
# pprint.pprint(db.list_runs()[:2])

# FIXME: this test was counting on the fact it's running after some test (I think test_httpdb) which leaves runs and
#  artifacts in the `results` dir, it should generate its own stuff, skipping for now
@pytest.mark.skip("FIX_ME")
def test_list_runs():

    db = get_db()
    runs = db.list_runs()
    assert runs, "empty runs result"

    html = runs.show(display=False)

    with open(f"{results}/runs.html", "w") as fp:
        fp.write(html)


# FIXME: this test was counting on the fact it's running after some test (I think test_httpdb) which leaves runs and
#  artifacts in the `results` dir, it should generate its own stuff, skipping for now
@pytest.mark.skip("FIX_ME")
def test_list_artifacts():

    db = get_db()
    artifacts = db.list_artifacts()
    assert artifacts, "empty artifacts result"

    html = artifacts.show(display=False)

    with open(f"{results}/artifacts.html", "w") as fp:
        fp.write(html)
