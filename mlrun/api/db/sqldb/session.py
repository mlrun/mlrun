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
import typing

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker as SessionMaker

from mlrun.config import config


def create_session(dsn=None) -> Session:
    session_maker = DBEngine(dsn=dsn)._get_session_maker()
    return session_maker()


class DBEngine:

    _engines: typing.Dict[str, Engine] = {}
    _session_makers: typing.Dict[str, SessionMaker] = {}

    def __init__(self, dsn=None):
        self._dsn = dsn or config.httpdb.dsn

    # doing lazy load to allow tests to initialize the engine
    def get_engine(self) -> Engine:
        if self._dsn not in self._engines:
            self._init_engine()
        return self._engines[self._dsn]

    # TODO: we accept the dsn here to enable tests to override it, the "right" thing will be that config will be easily
    #  overridable by tests (today when you import the config it is already being initialized.. should be lazy load)
    def _init_engine(self):
        kwargs = {}
        if "mysql" in self._dsn:
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
        engine = create_engine(self._dsn, **kwargs)
        self._engines[self._dsn] = engine
        self._init_session_maker()

    # doing lazy load to allow tests to initialize the engine
    def _get_session_maker(self) -> SessionMaker:
        if self._dsn not in self._session_makers:
            self._init_session_maker()
        return self._session_makers[self._dsn]

    def _init_session_maker(self):
        _session_maker = SessionMaker(bind=self.get_engine())
        self._session_makers[self._dsn] = _session_maker
