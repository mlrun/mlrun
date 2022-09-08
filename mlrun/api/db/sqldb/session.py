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
#
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker as SessionMaker

from mlrun.config import config

engine: Engine = None
_session_maker: SessionMaker = None


# doing lazy load to allow tests to initialize the engine
def get_engine() -> Engine:
    global engine
    if engine is None:
        _init_engine()
    return engine


def create_session() -> Session:
    session_maker = _get_session_maker()
    return session_maker()


# doing lazy load to allow tests to initialize the engine
def _get_session_maker() -> SessionMaker:
    global _session_maker
    if _session_maker is None:
        _init_session_maker()
    return _session_maker


# TODO: we accept the dsn here to enable tests to override it, the "right" thing will be that config will be easily
#  overridable by tests (today when you import the config it is already being initialized.. should be lazy load)
def _init_engine(dsn=None):
    global engine
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
        }
    engine = create_engine(dsn, **kwargs)
    _init_session_maker()


def _init_session_maker():
    global _session_maker
    _session_maker = SessionMaker(bind=get_engine())
