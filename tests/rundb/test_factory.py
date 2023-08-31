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
#
import unittest.mock

import pytest

import mlrun.db.factory
import mlrun.db.httpdb
import mlrun.db.nopdb


def test_create_http_db(monkeypatch):
    monkeypatch.setattr(
        mlrun.db.httpdb.HTTPRunDB,
        "connect",
        lambda *args, **kwargs: unittest.mock.Mock(),
    )

    factory = mlrun.db.factory.RunDBFactory()
    run_db = factory.create_run_db(url="https://fake-url")
    assert isinstance(run_db, mlrun.db.httpdb.HTTPRunDB)


def test_schema_validation():
    factory = mlrun.db.factory.RunDBFactory()
    with pytest.raises(ValueError):
        factory.create_run_db(url="not-https://localhost")


def test_create_nop_db():
    factory = mlrun.db.factory.RunDBFactory()
    run_db = factory.create_run_db(url="nop")
    assert isinstance(run_db, mlrun.db.nopdb.NopDB)
