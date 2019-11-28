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

from sqlalchemy import (
    BLOB, TIMESTAMP, Column, ForeignKey, Integer, String, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from ..lists import ArtifactList, FunctionList, RunList
from ..utils import get_in, update_in
from .base import RunDBError, RunDBInterface

Base = declarative_base()
NULL = None  # avoid flake8 issuing warnings when comparing in filter
run_time_fmt = '%Y-%m-%d %H:%M:%S.%f'


class HasStruct:
    @property
    def struct(self):
        return pickle.loads(self.body)

    @struct.setter
    def struct(self, value):
        self.body = pickle.dumps(value)


def make_label(table):
    class Label(Base):
        __tablename__ = f'{table}_labels'

        id = Column(Integer, primary_key=True)
        name = Column(String)
        parent = Column(Integer, ForeignKey(f'{table}.id'))

    return Label


class Artifact(Base, HasStruct):
    __tablename__ = 'artifacts'
    Label = make_label(__tablename__)

    id = Column(Integer, primary_key=True)
    key = Column(String)
    project = Column(String)
    tag = Column(String)
    uid = Column(String)
    updated = Column(TIMESTAMP)
    body = Column(BLOB)
    labels = relationship(Label)


class Function(Base, HasStruct):
    __tablename__ = 'functions'
    Label = make_label(__tablename__)

    id = Column(Integer, primary_key=True)
    name = Column(String)
    project = Column(String)
    tag = Column(String)
    body = Column(BLOB)
    labels = relationship(Label)


class Log(Base):
    __tablename__ = 'logs'

    id = Column(Integer, primary_key=True)
    uid = Column(String)
    project = Column(String)
    body = Column(BLOB)


class Run(Base, HasStruct):
    __tablename__ = 'runs'
    Label = make_label(__tablename__)

    id = Column(Integer, primary_key=True)
    uid = Column(String)
    project = Column(String)
    state = Column(String)
    body = Column(BLOB)
    start_time = Column(TIMESTAMP)
    labels = relationship(Label)


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

    def store_log(self, uid, project='', body=b'', append=True):
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
            state=run_state(struct),
            start_time=run_start_time(struct) or datetime.now(),
        )
        for label in run_labels(struct):
            run.labels.append(Run.Label(name=label, parent=run))
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
        new_state = run_state(struct)
        if new_state:
            run.state = new_state
        start_time = run_start_time(struct)
        if start_time:
            run.start_time = start_time
        run.labels.clear()
        for label in run_labels(struct):
            run.labels.append(Run.Label(name=label, parent=run))
        self._upsert(run)
        self._delete_empty_labels(Run.Label)

    def read_run(self, uid, project='', iter=0):
        run = self._query(Run, uid=uid, project=project).one_or_none()
        if not run:
            raise RunDBError(f'Run {uid}:{project} not found')
        return run.struct

    def list_runs(
            self, name='', uid=None, project='', labels=None,
            state='', sort=True, last=0, iter=False):

        query = self._find_runs(name, uid, project, labels, state)
        if sort:
            query = query.order_by(Run.start_time.desc())
        if last:
            query = query.limit(last)

        runs = RunList()
        for run in query:
            runs.append(run.struct)

        return runs

    def del_run(self, uid, project='', iter=0):
        self._delete(Run, uid=uid, project=project)

    def del_runs(self, name='', project='', labels=None, state='', days_ago=0):
        query = self._find_runs(name, '', project, labels, state)
        if days_ago:
            since = datetime.now() - timedelta(days=days_ago)
            query = query.filter(Run.start_time >= since)
        for run in query:  # Can't use query.delete with join
            self.session.delete(run)
        self.session.commit()

    def store_artifact(self, key, artifact, uid, tag='', project=''):
        artifact = artifact.copy()
        artifact['updated'] = datetime.now()
        art = Artifact(
            key=key,
            uid=uid,
            tag=tag,
            project=project)
        for label in label_set(artifact.get('labels', [])):
            art.labels.append(Artifact.Label(name=label, parent=art))
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
        arts.extend(
            obj.struct
            for obj in self._find_artifacts(name, project, tag, labels)
        )
        return arts

    def del_artifact(self, key, tag='', project=''):
        self._delete(
            Artifact, key=key, tag=tag, project=project)

    def del_artifacts(self, name='', project='', tag='', labels=None):
        for obj in self._find_artifacts(name, project, tag, labels):
            self.session.delete(obj)
        self.session.commit()

    def store_function(self, func, name, project='', tag=''):
        update_in(func, 'metadata.updated', datetime.now())
        fn = Function(
            name=name,
            project=project,
            tag=tag,
        )
        for label in label_set(get_in(func, 'metadata.labels', [])):
            fn.labels.append(Function.Label(name=label, parent=fn))
        fn.struct = func
        self._upsert(fn)

    def get_function(self, name, project='', tag=''):
        query = self._query(Function, name=name, project=project, tag=tag)
        obj = query.one_or_none()
        if obj:
            return obj.struct

    def list_functions(self, name, project='', tag='', labels=None):
        funcs = FunctionList()
        funcs.extend(
            obj.struct
            for obj in self._find_functions(name, project, tag, labels)
        )
        return funcs

    def _query(self, cls, **kw):
        kw = {k: v for k, v in kw.items() if v}
        return self.session.query(cls).filter_by(**kw)

    def _delete_empty_labels(self, cls):
        self.session.query(cls).filter(cls.parent == NULL).delete()
        self.session.commit()

    def _upsert(self, obj):
        self.session.add(obj)
        self.session.commit()

    def _find_runs(self, name, uid, project, labels, state):
        labels = label_set(labels)
        query = self._query(
            Run, name=name, uid=uid, project=project, state=state)
        if labels:
            query = query.join(Run.Label).filter(Run.Label.name.in_(labels))
        return query

    def _find_artifacts(self, name, project, tag, labels):
        # FIXME tag = tag or 'latest'
        labels = label_set(labels)
        query = self._query(Artifact, name=name, project=project, tag=tag)
        if labels:
            query = query.join(Run.Label).filter(Run.Label.name.in_(labels))
        return query

    def _find_functions(self, name, project, tag, labels):
        query = self._query(Function, name=name, project=project, tag=tag)
        labels = label_set(labels)

        for obj in query:
            func = obj.struct
            if labels:
                if not (labels & set(get_in(func, 'metadata.labels', []))):
                    continue
            yield obj

    def _delete(self, cls, **kw):
        query = self.session.query(cls).filter_by(**kw)
        for obj in query:
            self.session.delete(obj)
        self.session.commit()

    def _find_lables(self, cls, label_cls, labels):
        return self.session.query(cls).join(label_cls).filter(
                label_cls.name.in_(labels))


def label_set(labels):
    if isinstance(labels, str):
        labels = labels.split(',')

    return set(labels or [])


class RunWrapper:
    def __init__(self, run):
        self.run = run

    def __getattr__(self, attr):
        try:
            return self.run[attr]
        except KeyError:
            raise AttributeError(attr)


def run_start_time(run):
    ts = get_in(run, 'status.start_time', '')
    if not ts:
        return None
    return datetime.strptime(ts, run_time_fmt)


def run_labels(run):
    labels = get_in(run, 'metadata.labels', [])
    return label_set(labels)


def run_state(run):
    return get_in(run, 'status.state', '')
