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
from os import environ

from ..config import config
from .base import RunDBError, RunDBInterface  # noqa


def get_or_set_dburl(default=""):
    if not config.dbpath and default:
        config.dbpath = default
        environ["MLRUN_DBPATH"] = default
    return config.dbpath


def get_run_db(url="", secrets=None, force_reconnect=False) -> RunDBInterface:
    """Returns the runtime database"""
    # import here to avoid circular import
    import mlrun.db.factory

    run_db_factory = mlrun.db.factory.RunDBFactory()
    return run_db_factory.create_run_db(url, secrets, force_reconnect)
