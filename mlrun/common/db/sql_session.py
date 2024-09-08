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


from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import (
    sessionmaker as SessionMaker,  # noqa: N812 - `sessionmaker` is a class
)

from mlrun.config import config

# TODO: wrap the following functions in a singleton class
_engines: dict[str, Engine] = {}
_session_makers: dict[str, SessionMaker] = {}


# doing lazy load to allow tests to initialize the engine
def get_engine(dsn=None) -> Engine:
    global _engines
    dsn = dsn or config.httpdb.dsn
    if dsn not in _engines:
        _init_engine(dsn=dsn)
    return _engines[dsn]


def create_session(dsn=None) -> Session:
    session_maker = _get_session_maker(dsn=dsn)
    return session_maker()


# doing lazy load to allow tests to initialize the engine
def _get_session_maker(dsn) -> SessionMaker:
    global _session_makers
    dsn = dsn or config.httpdb.dsn
    if dsn not in _session_makers:
        _init_session_maker(dsn=dsn)
    return _session_makers[dsn]


# TODO: we accept the dsn here to enable tests to override it, the "right" thing will be that config will be easily
#  overridable by tests (today when you import the config it is already being initialized.. should be lazy load)
def _init_engine(dsn=None):
    global _engines
    dsn = dsn or config.httpdb.dsn
    kwargs = {}
    if "mysql" in dsn:
        pool_size = config.httpdb.db.connections_pool_size
        if pool_size is None:
            pool_size = config.httpdb.max_workers
        max_overflow = config.httpdb.db.connections_pool_max_overflow
        if max_overflow is None:
            max_overflow = config.httpdb.max_workers

        kwargs = {
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_pre_ping": config.httpdb.db.connections_pool_pre_ping,
            "pool_recycle": config.httpdb.db.connections_pool_recycle,
        }
    engine = create_engine(dsn, **kwargs)
    _engines[dsn] = engine
    _init_session_maker(dsn=dsn)


def _init_session_maker(dsn):
    global _session_makers
    _session_makers[dsn] = SessionMaker(bind=get_engine(dsn=dsn))
