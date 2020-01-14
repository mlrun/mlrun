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
import warnings

from sqlalchemy import (
    BLOB, TIMESTAMP, Column, ForeignKey, Integer, String, UniqueConstraint,
    create_engine, func
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from ..config import config
from ..lists import ArtifactList, FunctionList, RunList
from ..utils import get_in, update_in
from .base import RunDBError, RunDBInterface

Base = declarative_base()
NULL = None  # Avoid flake8 issuing warnings when comparing in filter
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


# quell SQLAlchemy warnings on duplicate class name (Label)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    class Artifact(Base, HasStruct):
        __tablename__ = 'artifacts'
        __table_args__ = (
            UniqueConstraint('uid', 'project', 'key', name='_artifacts_uc'),
        )

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
        __table_args__ = (
            UniqueConstraint('name', 'project', 'tag', name='_functions_uc'),
        )

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
        __table_args__ = (
            UniqueConstraint('uid', 'project', 'iteration', name='_runs_uc'),
        )

        Label = make_label(__tablename__)

        id = Column(Integer, primary_key=True)
        uid = Column(String)
        project = Column(String)
        iteration = Column(Integer)
        state = Column(String)
        body = Column(BLOB)
        start_time = Column(TIMESTAMP)
        labels = relationship(Label)

    class Schedule(Base, HasStruct):
        __tablename__ = 'schedules'

        id = Column(Integer, primary_key=True)
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

    def store_log(self, uid, project='', body=b'', append=False):
        project = project or config.default_project
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
        project = project or config.default_project
        log = self._query(Log, uid=uid, project=project).one_or_none()
        if not log:
            return None, None
        end = None if size == 0 else offset + size
        return '', log.body[offset:end]

    def store_run(self, struct, uid, project='', iter=0):
        project = project or config.default_project
        run = self._get_run(uid, project, iter)
        if not run:
            run = Run(
                uid=uid,
                project=project,
                iteration=iter,
                state=run_state(struct),
                start_time=run_start_time(struct) or datetime.now(),
            )
        labels = run_labels(struct)
        update_labels(run, labels)
        run.struct = struct
        self._upsert(run)

    def update_run(self, updates: dict, uid, project='', iter=0):
        project = project or config.default_project
        run = self._get_run(uid, project, iter)
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
        self.session.merge(run)
        self.session.commit()
        self._delete_empty_labels(Run.Label)

    def read_run(self, uid, project='', iter=0):
        project = project or config.default_project
        run = self._get_run(uid, project, iter)
        if not run:
            raise RunDBError(f'Run {uid}:{project} not found')
        return run.struct

    def list_runs(
            self, name='', uid=None, project='', labels=None,
            state='', sort=True, last=0, iter=False):
        project = project or config.default_project
        query = self._find_runs(name, uid, project, labels, state)
        if sort:
            query = query.order_by(Run.start_time.desc())
        if last:
            query = query.limit(last)
        if not iter:
            query = query.filter(Run.iteration == 0)

        runs = RunList()
        for run in query:
            runs.append(run.struct)

        return runs

    def del_run(self, uid, project='', iter=0):
        project = project or config.default_project
        # We currently delete *all* iterations
        self._delete(Run, uid=uid, project=project)

    def del_runs(
        self, name='', project='', labels=None,
            state='', days_ago=0):
        project = project or config.default_project
        query = self._find_runs(name, '', project, labels, state)
        if days_ago:
            since = datetime.now() - timedelta(days=days_ago)
            query = query.filter(Run.start_time >= since)
        for run in query:  # Can't use query.delete with join
            self.session.delete(run)
        self.session.commit()

    def store_artifact(
            self, key, artifact, uid, tag='', project=''):
        project = project or config.default_project
        artifact = artifact.copy()
        updated = artifact['updated'] = datetime.now()
        art = self._get_artifact(uid, project, key)
        labels = label_set(artifact.get('labels', []))
        if not art:
            art = Artifact(
                key=key,
                uid=uid,
                tag=tag,
                updated=updated,
                project=project)
        update_labels(art, labels)
        art.struct = artifact
        self._upsert(art)

    def read_artifact(self, key, tag='', project=''):
        project = project or config.default_project
        query = self._query(
            Artifact, key=key, project=project)

        if tag:
            query = query.filter(Artifact.uid == tag)
        else:
            # Select by last updated
            max_updated = self.session.query(
                func.max(Artifact.updated)).filter(
                    Artifact.project == project, Artifact.key == key)
            query = query.filter(Artifact.updated.in_(max_updated))

        art = query.one_or_none()
        if not art:
            raise RunDBError(f'Artifact {key}:{tag}:{project} not found')
        return art.struct

    def list_artifacts(
            self, name='', project='', tag='', labels=None):
        project = project or config.default_project
        arts = ArtifactList()
        arts.extend(
            obj.struct
            for obj in self._find_artifacts(name, project, tag, labels)
        )
        return arts

    def del_artifact(self, key, tag='', project=''):
        project = project or config.default_project
        self._delete(
            Artifact, key=key, tag=tag, project=project)

    def del_artifacts(
            self, name='', project='', tag='', labels=None):
        project = project or config.default_project
        for obj in self._find_artifacts(name, project, tag, labels):
            self.session.delete(obj)
        self.session.commit()

    def store_function(self, func, name, project='', tag=''):
        project = project or config.default_project
        update_in(func, 'metadata.updated', datetime.now())
        fn = self._get_function(name, project, tag)
        if not fn:
            fn = Function(
                name=name,
                project=project,
                tag=tag,
            )
        labels = label_set(get_in(func, 'metadata.labels', []))
        update_labels(fn, labels)
        fn.struct = func
        self._upsert(fn)

    def get_function(self, name, project='', tag=''):
        project = project or config.default_project
        query = self._query(Function, name=name, project=project, tag=tag)
        obj = query.one_or_none()
        if obj:
            return obj.struct

    def list_functions(
            self, name, project='', tag='', labels=None):
        project = project or config.default_project
        funcs = FunctionList()
        funcs.extend(
            obj.struct
            for obj in self._find_functions(name, project, tag, labels)
        )
        return funcs

    def list_projects(self):
        return [row[0] for row in self.session.query(Run.project).distinct()]

    def list_artifact_tags(self, project):
        query = self.session.query(Artifact.tag).filter(
            Artifact.project == project).distinct()
        return [row[0] for row in query]

    def store_schedule(self, data):
        sched = Schedule()
        sched.struct = data
        self._upsert(sched)

    def list_schedules(self):
        return [s.struct for s in self.session.query(Schedule)]

    def _query(self, cls, **kw):
        kw = {k: v for k, v in kw.items() if v}
        return self.session.query(cls).filter_by(**kw)

    def _get_function(self, name, project, tag):
        query = self._query(Function, name=name, project=project, tag=tag)
        return query.one_or_none()

    def _get_artifact(self, uid, project, key):
        query = self._query(Artifact, uid=uid, project=project, key=key)
        return query.one_or_none()

    def _get_run(self, uid, project, iteration):
        return self._query(
            Run, uid=uid, project=project, iteration=iteration).one_or_none()

    def _delete_empty_labels(self, cls):
        self.session.query(cls).filter(cls.parent == NULL).delete()
        self.session.commit()

    def _upsert(self, obj):
        try:
            self.session.add(obj)
            self.session.commit()
        except SQLAlchemyError as err:
            self.session.rollback()
            cls = obj.__class__.__name__
            raise RunDBError(f'duplicate {cls} - {err}')

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


def update_labels(obj, labels):
    old = {label.name: label for label in obj.labels}
    obj.labels.clear()
    for name in labels:
        if name in old:
            obj.labels.append(old[name])
        else:
            obj.labels.append(obj.Label(name=name, parent=obj))
