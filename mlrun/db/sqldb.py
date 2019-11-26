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

import pickle
from datetime import datetime, timedelta

from sqlalchemy import BLOB, Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..lists import RunList, ArtifactList, FunctionList
from ..utils import get_in, update_in
from .base import RunDBError, RunDBInterface

Base = declarative_base()


class HasStruct:
    @property
    def struct(self):
        return pickle.loads(self.body)

    @struct.setter
    def struct(self, value):
        self.body = pickle.dumps(value)


# TODO: Store labels in db (one to many)
class Artifact(Base, HasStruct):
    __tablename__ = 'artifacts'

    id = Column(Integer, primary_key=True)
    key = Column(String)
    project = Column(String)
    tag = Column(String)
    uid = Column(String)
    body = Column(BLOB)


class Function(Base, HasStruct):
    __tablename__ = 'functions'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    project = Column(String)
    tag = Column(String)
    body = Column(BLOB)


class Log(Base):
    __tablename__ = 'logs'

    id = Column(Integer, primary_key=True)
    uid = Column(String)
    project = Column(String)
    body = Column(BLOB)


class Run(Base, HasStruct):
    __tablename__ = 'runs'
    id = Column(Integer, primary_key=True)
    uid = Column(String)
    project = Column(String)
    body = Column(BLOB)


class SQLDB(RunDBInterface):
    def __init__(self, dsn):
        self.dsn = dsn
        self.session = None

    def connect(self, secrets=None):
        engine = create_engine(self.dsn)
        Base.metadata.create_all(engine)
        cls = sessionmaker(bind=engine)
        # TODO: One session per call?
        self.session = cls()
        return self

    def store_log(self, uid, project='', body=None, append=True):
        log = self._query(Log, uid=uid, project=project).one_or_none()
        if not log:
            log = Log(uid=uid, project=project, body=body)
        elif body:
            if append:
                log.body += body
            else:
                log.body = body
        self._upsert(log)

    def get_log(self, uid, project='', offset=0, size=0):
        log = self._query(Log, uid=uid, project=project).one_or_none()
        if not log:
            return None
        end = None if size == 0 else offset + size
        return log.body[offset:end]

    def store_run(self, struct, uid, project='', iter=0):
        run = Run(
            uid=uid,
            project=project,
        )
        run.struct = struct
        self._upsert(run)

    def update_run(self, updates: dict, uid, project='', iter=0):
        run = self._query(Run, uid=uid, project=project).one_or_none()
        if not run:
            raise RunDBError(f'run {uid}:{project} not found')
        struct = run.struct
        for key, val in updates.items():
            update_in(struct, key, val)
        run.struct = struct
        self._upsert(run)

    def read_run(self, uid, project='', iter=0):
        run = self._query(Run, uid=uid, project=project).one_or_none()
        if not run:
            raise RunDBError(f'Run {uid}:{project} not found')
        return run.struct

    def list_runs(
            self, name='', uid=None, project='', labels=None,
            state='', sort=True, last=0, iter=False):
        runs = RunList()
        for run in self._iter_runs(name, uid, project, labels, state):
            runs.append(run.struct)

        if sort or last:
            runs.sort(key=_run_start, reverse=True)
        if last:
            runs = runs[:last]
        return run

    def del_run(self, uid, project='', iter=0):
        self._delete(Run, uid=uid, project=project)

    def del_runs(self, name='', project='', labels=None, state='', days_ago=0):
        labels = _label_set(labels)
        query = self.session.query(Run)
        if name:
            query = query.filter(Run.name == name)
        if project:
            query = query.filter(Run.project == project)

        if days_ago:
            since = datetime.now() - timedelta(days=days_ago)

        def start_ok(struct):
            ts = get_in(struct, 'status.start_time')
            time = datetime.strptime('%Y-%m-%d %H:%M:%S.%f', ts)
            return time >= since

        for run in query:
            if not self._match_run(run, labels, state):
                continue
            if not start_ok(run.struct):
                continue
            self.session.delete(run)
        self.session.commit()

    def store_artifact(self, key, artifact, uid, tag='', project=''):
        art = Artifact(
            key=key,
            uid=uid,
            tag=tag,
            project=project)
        art.struct = artifact
        self._upsert(art)

    def read_artifact(self, key, tag='', project=''):
        art = self._query(
            Artifact, key=key, tag=tag, project=project).one_or_none()
        if not art:
            raise RunDBError(f'Artifact {key}:{tag}:{project} not found')
        return art.struct

    def list_artifacts(self, name='', project='', tag='', labels=None):
        arts = ArtifactList()

        for obj in self._iter_artifcats(name, project, tag, labels):
            arts.append(obj.struct)

        return arts

    def del_artifact(self, key, tag='', project=''):
        self._delete(
            Artifact, key=key, tag=tag, project=project)

    def del_artifacts(self, name='', project='', tag='', labels=None):
        for obj in self._iter_artifcats(name, project, tag, labels):
            self.session.delete(obj)
        self.session.commit()

    def store_function(self, func, name, project='', tag=''):
        fn = Function(
            name=name,
            project=project,
            tag=tag,
        )
        fn.struct = func
        self._upsert(fn)

    def get_function(self, name, project='', tag=''):
        query = self._query(Function, name=name, project=project, tag=tag)
        obj = query.one_or_none()
        if obj:
            return obj.struct

    def list_functions(self, name, project='', tag='', labels=None):
        return FunctionList(
            obj.struct
            for obj in self._iter_functions(name, project, tag, labels)
        )

    def _query(self, cls, **kw):
        kw = {k: v for k, v in kw.items() if v}
        return self.session.query(cls).filter_by(**kw)

    def _upsert(self, obj):
        self.session.add(obj)
        self.session.commit()

    def _match_run(self, run, labels, state):
        meta = run.struct.get('metadata', {})
        if state and meta.get('state') != state:
            return False
        if labels:
            run_labels = set(meta.get('labels', []))
            if not labels & run_labels:
                return False
        return True

    def _iter_runs(self, name, uid, project, labels, state):
        labels = _label_set(labels)
        query = self._uid_prj_query(Run, uid, project)
        for run in query:
            if self._match_run(run, labels, state):
                yield run

    def _iter_artifcats(self, name, project, tag, labels):
        tag = tag or 'latest'
        query = self._query(Artifact, name=name, project=project, tag=tag)
        labels = _label_set(labels)

        for obj in query:
            art = obj.struct
            if not (labels & set(art.get('labels', []))):
                continue
            yield obj

    def _iter_functions(self, name, project, tag, labels):
        query = self._query(Function, name=name, project=project, tag=tag)
        labels = _label_set(labels)

        for obj in query:
            func = obj.struct
            if not (labels & set(get_in(func, 'metadata.labels', []))):
                continue
            yield obj

    def _delete(self, cls, **kw):
        query = self.session.query(cls).filter_by(**kw)
        for obj in query:
            self.session.delete(obj)
        self.session.commit()


def _label_set(labels):
    if isinstance(labels, str):
        labels = labels.split(',')

    return set(labels or [])


def _run_start(run):
    return get_in(run, ['status', 'start_time'], '')
