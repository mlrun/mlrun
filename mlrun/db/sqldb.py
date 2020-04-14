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
import warnings
from datetime import datetime, timedelta, timezone

from dateutil import parser
from sqlalchemy import (
    BLOB, TIMESTAMP, Column, ForeignKey, Integer, String, Table,
    UniqueConstraint, and_, create_engine, func
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from ..config import config
from ..lists import ArtifactList, FunctionList, RunList
from ..utils import get_in, update_in, logger
from .base import RunDBError, RunDBInterface

from threading import RLock

sql_lock = RLock()
Base = declarative_base()
NULL = None  # Avoid flake8 issuing warnings when comparing in filter
run_time_fmt = '%Y-%m-%dT%H:%M:%S.%fZ'


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
        __table_args__ = (
            UniqueConstraint('name', 'parent', name=f'_{table}_labels_uc'),
        )

        id = Column(Integer, primary_key=True)
        name = Column(String)
        value = Column(String)
        parent = Column(Integer, ForeignKey(f'{table}.id'))

    return Label


def make_tag(table):
    class Tag(Base):
        __tablename__ = f'{table}_tags'
        __table_args__ = (
            UniqueConstraint(
                'project', 'name', 'obj_id', name=f'_{table}_tags_uc'),
        )

        id = Column(Integer, primary_key=True)
        project = Column(String)
        name = Column(String)
        obj_id = Column(Integer, ForeignKey(f'{table}.id'))

    return Tag


# quell SQLAlchemy warnings on duplicate class name (Label)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    class Artifact(Base, HasStruct):
        __tablename__ = 'artifacts'
        __table_args__ = (
            UniqueConstraint('uid', 'project', 'key', name='_artifacts_uc'),
        )

        Label = make_label(__tablename__)
        Tag = make_tag(__tablename__)

        id = Column(Integer, primary_key=True)
        key = Column(String)
        project = Column(String)
        uid = Column(String)
        updated = Column(TIMESTAMP)
        body = Column(BLOB)
        labels = relationship(Label)

    class Function(Base, HasStruct):
        __tablename__ = 'functions'
        __table_args__ = (
            UniqueConstraint('name', 'project', 'uid', name='_functions_uc'),
        )

        Label = make_label(__tablename__)
        Tag = make_tag(__tablename__)

        id = Column(Integer, primary_key=True)
        name = Column(String)
        project = Column(String)
        uid = Column(String)
        body = Column(BLOB)
        updated = Column(TIMESTAMP)
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
        Tag = make_tag(__tablename__)

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

    # Define "many to many" users/projects
    project_users = Table(
        'project_users', Base.metadata,
        Column('project_id', Integer, ForeignKey('projects.id')),
        Column('user_id', Integer, ForeignKey('users.id')),
    )

    class User(Base):
        __tablename__ = 'users'
        __table_args__ = (
            UniqueConstraint('name', name='_users_uc'),
        )

        id = Column(Integer, primary_key=True)
        name = Column(String)

    class Project(Base):
        __tablename__ = 'projects'
        # For now since we use project name a lot
        __table_args__ = (
            UniqueConstraint('name', name='_projects_uc'),
        )

        id = Column(Integer, primary_key=True)
        name = Column(String)
        description = Column(String)
        owner = Column(String)
        source = Column(String)
        _spec = Column('spec', BLOB)
        created = Column(TIMESTAMP, default=datetime.utcnow)
        state = Column(String)
        users = relationship(User, secondary=project_users)

        @property
        def spec(self):
            if self._spec:
                return pickle.loads(self._spec)

        @spec.setter
        def spec(self, value):
            self._spec = pickle.dumps(value)


# Must be after all table definitions
_tagged = [cls for cls in Base.__subclasses__() if hasattr(cls, 'Tag')]
_table2cls = {cls.__table__.name: cls for cls in Base.__subclasses__()}


class SQLDB(RunDBInterface):
    def __init__(self, dsn):
        self.dsn = dsn
        self.session = None
        self._projects = set()  # project cache

    def connect(self, secrets=None):
        engine = create_engine(self.dsn)
        Base.metadata.create_all(engine)
        cls = sessionmaker(bind=engine)
        # TODO: One session per call?
        self.session = cls()

        for project in self.list_projects():
            self._projects.add(project.name)

    def store_log(self, uid, project='', body=b'', append=False):
        project = project or config.default_project
        self._create_project_if_not_exists(project)
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
        self._create_project_if_not_exists(project)
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
        self._upsert(run, ignore=True)

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
        for name, value in run_labels(struct).items():
            lbl = Run.Label(name=name, value=value, parent=run.id)
            run.labels.append(lbl)
        self.session.merge(run)
        self.session.commit()
        self._delete_empty_labels(Run.Label)

    def read_run(self, uid, project=None, iter=None):
        project = project or config.default_project
        run = self._get_run(uid, project, iter)
        if not run:
            raise RunDBError(f'Run {uid}:{project} not found')
        return run.struct

    def list_runs(
            self, name=None, uid=None, project=None, labels=None,
            state=None, sort=True, last=0, iter=None):
        # FIXME: Run has no "name"
        project = project or config.default_project
        query = self._find_runs(uid, project, labels, state)
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

    def del_run(self, uid, project=None, iter=None):
        project = project or config.default_project
        # We currently delete *all* iterations
        self._delete(Run, uid=uid, project=project)

    def del_runs(
        self, name=None, project=None, labels=None,
            state=None, days_ago=0):
        # FIXME: Run has no `name`
        project = project or config.default_project
        query = self._find_runs(None, project, labels, state)
        if days_ago:
            since = datetime.now() - timedelta(days=days_ago)
            query = query.filter(Run.start_time >= since)
        for run in query:  # Can't use query.delete with join
            self.session.delete(run)
        self.session.commit()

    def store_artifact(
            self, key, artifact, uid, iter=None, tag='', project=''):
        project = project or config.default_project
        self._create_project_if_not_exists(project)
        artifact = artifact.copy()
        updated = artifact.get('updated')
        if not updated:
            updated = artifact['updated'] = datetime.now(timezone.utc)
        if iter:
            key = '{}-{}'.format(iter, key)
        art = self._get_artifact(uid, project, key)
        labels = artifact.get('labels', {})
        if not art:
            art = Artifact(
                key=key,
                uid=uid,
                updated=updated,
                project=project)
        update_labels(art, labels)
        art.struct = artifact
        self._upsert(art)
        if tag:
            self.tag_objects([art], project, tag)

    def read_artifact(self, key, tag='', iter=None, project=''):
        project = project or config.default_project
        uid = self._resolve_tag(Artifact, project, tag)
        if iter:
            key = '{}-{}'.format(iter, key)

        query = self._query(
            Artifact, key=key, project=project)
        if uid:
            query = query.filter(Artifact.uid == uid)
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
        self, name=None, project=None, tag=None, labels=None,
            since=None, until=None):
        project = project or config.default_project
        uid = 'latest'
        if tag:
            uid = self._resolve_tag(Artifact, project, tag)

        arts = ArtifactList(
            obj.struct
            for obj in self._find_artifacts(project, uid, labels, since, until)
        )
        return arts

    def del_artifact(self, key, tag='', project=''):
        project = project or config.default_project
        kw = {
            'key': key,
            'project': project,
        }
        if tag:
            kw['tag'] = tag

        self._delete(Artifact, **kw)

    def del_artifacts(
            self, name='', project='', tag='', labels=None):
        project = project or config.default_project
        for obj in self._find_artifacts(project, tag, labels, None, None):
            self.session.delete(obj)
        self.session.commit()

    def store_function(self, func, name, project='', tag=''):
        project = project or config.default_project
        self._create_project_if_not_exists(project)
        tag = tag or get_in(func, 'metadata.tag') or 'latest'

        #uid = self._resolve_tag(Function, project, tag)
        updated = datetime.now(timezone.utc)
        update_in(func, 'metadata.updated', updated)
#        fn = self._get_function(name, project, uid)
        fn = self._get_function(name, project, tag)
        if not fn:
            fn = Function(
                name=name,
                project=project,
                uid=tag,
            )
        fn.updated = updated
        labels = get_in(func, 'metadata.labels', {})
        update_labels(fn, labels)
        fn.struct = func
        self._upsert(fn)

    def get_function(self, name, project='', tag=''):
        project = project or config.default_project
        query = self._query(Function, name=name, project=project)
        tag = tag or 'latest'
        # if tag:
        #     if tag == 'latest':
        #         uid = self._function_latest_uid(project, name)
        #         if not uid:
        #             raise RunDBError(
        #                 f'no latest version for function {project}:{name}')
        #     else:
        #         uid = self._resolve_tag(Function, project, tag)
        query = query.filter(Function.uid == tag)
        obj = query.one_or_none()
        if obj:
            return obj.struct

    def list_functions(
            self, name, project=None, tag=None, labels=None):
        project = project or config.default_project
        uid = tag
        # if uid:
        #     uid = self._resolve_tag(Function, project, uid)

        funcs = FunctionList()
        funcs.extend(
            obj.struct
            for obj in self._find_functions(name, project, uid, labels)
        )
        return funcs

    def list_artifact_tags(self, project):
        query = self.session.query(Artifact.Tag.name).filter(
            Artifact.Tag.project == project).distinct()
        return [row[0] for row in query]

    def store_schedule(self, data):
        sched = Schedule()
        sched.struct = data
        self._upsert(sched)

    def list_schedules(self):
        return [s.struct for s in self.session.query(Schedule)]

    def tag_objects(self, objs, project: str, name: str):
        """Tag objects with (project, name) tag.

        If force==True will update tag
        """
        for obj in objs:
            tag = obj.Tag(project=project, name=name, obj_id=obj.id)
            self.session.add(tag)

    def del_tag(self, project: str, name: str):
        """Remove tag (project, name) from all objects"""
        count = 0
        for cls in _tagged:
            for obj in self._query(cls.Tag, project=project, name=name):
                self.session.delete(obj)
                count += 1
        return count

    def find_tagged(self, project: str, name: str):
        """Return all objects tagged with (project, name)

        If not tag found, will return an empty str.
        """
        objs = []
        for cls in _tagged:
            for tag in self._query(cls.Tag, project=project, name=name):
                objs.append(self._query(cls).get(tag.obj_id))
        return objs

    def list_tags(self, project: str):
        """Return all tags for a project"""
        tags = set()
        for cls in _tagged:
            for tag in self._query(cls.Tag, project=project):
                tags.add(tag.name)
        return tags

    def add_project(self, project: dict):
        project = project.copy()
        name = project.get('name')
        if not name:
            raise ValueError('project missing name')

        user_names = project.pop('users', [])
        prj = Project(**project)
        users = [] #self._find_or_create_users(user_names)
        prj.users.extend(users)
        self._upsert(prj)
        self._projects.add(prj.name)
        return prj.id

    def update_project(self, name, data: dict):
        prj = self.get_project(name)
        if not prj:
            raise RunDBError(f'unknown project - {name}')

        data = data.copy()
        user_names = data.pop('users', [])
        for key, value in data.items():
            if not hasattr(prj, key):
                raise RunDBError(f'unknown project attribute - {key}')
            setattr(prj, key, value)

        users = [] #self._find_or_create_users(user_names)
        prj.users.clear()
        prj.users.extend(users)
        self._upsert(prj, ignore=True)

    def get_project(self, name=None, project_id=None):
        if not (project_id or name):
            raise ValueError('need at least one of project_id or name')

        if project_id:
            return self._query(Project).get(project_id)

        return self._query(Project, name=name).one_or_none()

    def list_projects(self, owner=None):
        return self._query(Project, owner=owner)

    def _resolve_tag(self, cls, project, name):
        for tag in self._query(cls.Tag, project=project, name=name):
            return self._query(cls).get(tag.obj_id).uid
        return name  # Not found, return original uid

    def _query(self, cls, **kw):
        kw = {k: v for k, v in kw.items() if v is not None}
        return self.session.query(cls).filter_by(**kw)

    def _function_latest_uid(self, project, name):
        # FIXME
        query = (
            self._query(Function.uid)
            .filter(Function.project == project, Function.name == name)
            .order_by(Function.updated.desc())
        ).limit(1)
        out = query.one_or_none()
        if out:
            return out[0]

    def _create_project_if_not_exists(self, name):
        if name not in self._projects:
            self.add_project({'name': name})

    def _find_or_create_users(self, user_names):
        users = list(self._query(User).filter(User.name.in_(user_names)))
        new = set(user_names) - {user.name for user in users}
        if new:
            for name in new:
                user = User(name=name)
                self.session.add(user)
                users.append(user)
            try:
                self.session.commit()
            except SQLAlchemyError as err:
                self.session.rollback()
                raise RunDBError(f'add user: {err}') from err
        return users

    def _get_function(self, name, project, tag):
        uid = self._resolve_tag(Function, project, tag)
        query = self._query(Function, name=name, project=project, uid=uid)
        return query.one_or_none()

    def _get_artifact(self, uid, project, key):
        try:
            sql_lock.acquire()
            resp = self._query(
                Artifact, uid=uid, project=project, key=key).one_or_none()
            return resp
        finally:
            sql_lock.release()
            pass

    def _get_run(self, uid, project, iteration):
        try:
            sql_lock.acquire()
            resp = self._query(
                Run, uid=uid, project=project, iteration=iteration).one_or_none()
            return resp
        finally:
            sql_lock.release()
            pass

    def _delete_empty_labels(self, cls):
        self.session.query(cls).filter(cls.parent == NULL).delete()
        self.session.commit()

    def _upsert(self, obj, ignore=False):
        try:
            sql_lock.acquire()
            self.session.add(obj)
            self.session.commit()
        except SQLAlchemyError as err:
            self.session.rollback()
            cls = obj.__class__.__name__
            logger.warning(f'conflict adding {cls}, {err}')
            if not ignore:
                sql_lock.release()
                raise RunDBError(f'duplicate {cls} - {err}') from err
        sql_lock.release()

    def _find_runs(self, uid, project, labels, state):
        labels = label_set(labels)
        query = self._query(Run, uid=uid, project=project, state=state)
        return self._add_labels_filter(query, Run, labels)

    def _latest_uid_filter(self, query):
        # Create a sub query of latest uid (by updated) per (project,key)
        subq = self.session.query(
            Artifact.uid,
            Artifact.project,
            Artifact.key,
            func.max(Artifact.updated),
        ).group_by(
            Artifact.project,
            Artifact.key.label('key'),
        ).subquery('max_key')

        # Join curreny query with sub query on (project, key, uid)
        return query.join(
            subq,
            and_(
                Artifact.project == subq.c.project,
                Artifact.key == subq.c.key,
                Artifact.uid == subq.c.uid,
            )
        )

    def _find_artifacts(self, project, uid, labels, since, until):
        labels = label_set(labels)
        query = self._query(Artifact, project=project)
        if uid != '*':
            if uid == 'latest':
                query = self._latest_uid_filter(query)
            else:
                query = query.filter(Artifact.uid == uid)

        query = self._add_labels_filter(query, Artifact, labels)

        if since or until:
            since = since or datetime.min
            until = until or datetime.max
            query = query.filter(and_(
                Artifact.updated >= since,
                Artifact.updated <= until
            ))

        return query

    def _find_functions(self, name, project, uid, labels):
        query = self._query(Function, name=name, project=project)
        if uid:
            query = query.filter(Function.uid == uid)

        labels = label_set(labels)
        return self._add_labels_filter(query, Function, labels)

    def _delete(self, cls, **kw):
        query = self.session.query(cls).filter_by(**kw)
        for obj in query:
            self.session.delete(obj)
        self.session.commit()

    def _find_lables(self, cls, label_cls, labels):
        return self.session.query(cls).join(label_cls).filter(
                label_cls.name.in_(labels))

    def _add_labels_filter(self, query, cls, labels):
        if not labels:
            return query

        preds = []
        for lbl in labels:
            if '=' in lbl:
                name, value = [v.strip() for v in lbl.split('=', 1)]
                cond = and_(cls.Label.name == name, cls.Label.value == value)
                preds.append(cond)
            else:
                preds.append(cls.Label.name == lbl.strip())

        subq = self.session.query(cls.Label).filter(*preds).subquery('labels')
        return query.join(subq)


def table2cls(name):
    return _table2cls.get(name)


def label_set(labels):
    if isinstance(labels, str):
        labels = labels.split(',')

    return set(labels or [])


def run_start_time(run):
    ts = get_in(run, 'status.start_time', '')
    if not ts:
        return None
    return parser.parse(ts)


def run_labels(run) -> dict:
    return get_in(run, 'metadata.labels', {})


def run_state(run):
    return get_in(run, 'status.state', '')


def update_labels(obj, labels: dict):
    old = {label.name: label for label in obj.labels}
    obj.labels.clear()
    for name, value in labels.items():
        if name in old:
            obj.labels.append(old[name])
        else:
            obj.labels.append(obj.Label(name=name, value=value, parent=obj.id))


def to_dict(obj):
    if isinstance(obj, Base):
        return {
            attr: to_dict(getattr(obj, attr))
            for attr in dir(obj)
            if is_field(attr)
        }

    if isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(to_dict(v) for v in obj)

    return obj


def is_field(name):
    if name[0] == '_':
        return False
    return name not in ('metadata', 'Tag', 'Label')
