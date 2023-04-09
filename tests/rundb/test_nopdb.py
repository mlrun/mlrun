# Copyright 2022 Iguazio
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


def test_nopdb():
    # by default we use a nopdb with raise_error = False
    assert mlrun.mlconf.httpdb.nop_db.raise_error is False

    rundb = mlrun.get_run_db()
    assert isinstance(rundb, mlrun.db.NopDB)

    # not expected to fail as it in the white list
    rundb.connect()

    # not expected to fail
    rundb.read_run("123")

    # set raise_error to True
    mlrun.mlconf.httpdb.nop_db.raise_error = True

    assert mlrun.mlconf.httpdb.nop_db.raise_error is True

    # not expected to fail as it in the white list
    rundb.connect()

    # expected to fail
    with pytest.raises(mlrun.errors.MLRunBadRequestError):
        rundb.read_run("123")
