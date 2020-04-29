from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    and_, func
)
from sqlalchemy.exc import SQLAlchemyError

from mlrun.app.db.base import DBError, DBInterface
from mlrun.app.db.sqldb.helpers import label_set, run_start_time, run_labels, run_state, update_labels
from mlrun.app.db.sqldb.models import Artifact, Function, Log, Run, Schedule, User, Project, _tagged
from mlrun.config import config
from mlrun.lists import ArtifactList, FunctionList, RunList
from mlrun.utils import get_in, update_in, logger

NULL = None  # Avoid flake8 issuing warnings when comparing in filter
run_time_fmt = "%Y-%m-%dT%H:%M:%S.%fZ"


class SQLDB(DBInterface):
    def __init__(self, dsn, projects=None):
        self.dsn = dsn

        # FIXME: this is a huge hack - the cache is currently not a real cache but a replica of the DB - if the code
        # can not find a project in the cache it will just try to create it instead of trying to get it from the db
        # first. in some cases we need several instances of this class (see mlrun.db.sqldb) therefore they all need to
        # have the same cache, receiving the cache here (set in python passed by reference) to enable that
        self._projects = projects or set()  # project cache

    def initialize(self, session):
        self._initialize_cache(session)

    def _initialize_cache(self, session):
        for project in self.list_projects(session):
            self._projects.add(project.name)

    def get_projects_cache(self):
        return self._projects

    def store_log(self, session, uid, project="", body=b"", append=False):
        project = project or config.default_project
        self._create_project_if_not_exists(session, project)
        log = self._query(session, Log, uid=uid, project=project).one_or_none()
        if not log:
            log = Log(uid=uid, project=project, body=body)
        elif body:
            if append:
                log.body += body
            else:
                log.body = body
        self._upsert(session, log)

    def get_log(self, session, uid, project="", offset=0, size=0):
        project = project or config.default_project
        log = self._query(session, Log, uid=uid, project=project).one_or_none()
        if not log:
            return None, None
        end = None if size == 0 else offset + size
        return "", log.body[offset:end]

    def store_run(self, session, struct, uid, project="", iter=0):
        project = project or config.default_project
        self._create_project_if_not_exists(session, project)
        run = self._get_run(session, uid, project, iter)
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
        self._upsert(session, run, ignore=True)

    def update_run(self, session, updates: dict, uid, project="", iter=0):
        project = project or config.default_project
        run = self._get_run(session, uid, project, iter)
        if not run:
            raise DBError(f"run {uid}:{project} not found")
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
        session.merge(run)
        session.commit()
        self._delete_empty_labels(session, Run.Label)

    def read_run(self, session, uid, project=None, iter=None):
        project = project or config.default_project
        run = self._get_run(session, uid, project, iter)
        if not run:
            raise DBError(f"Run {uid}:{project} not found")
        return run.struct

    def list_runs(
            self, session, name=None, uid=None, project=None, labels=None,
            state=None, sort=True, last=0, iter=None):
        # FIXME: Run has no "name"
        project = project or config.default_project
        query = self._find_runs(session, uid, project, labels, state)
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

    def del_run(self, session, uid, project=None, iter=None):
        project = project or config.default_project
        # We currently delete *all* iterations
        self._delete(session, Run, uid=uid, project=project)

    def del_runs(
            self, session, name=None, project=None, labels=None,
            state=None, days_ago=0):
        # FIXME: Run has no `name`
        project = project or config.default_project
        query = self._find_runs(session, None, project, labels, state)
        if days_ago:
            since = datetime.now() - timedelta(days=days_ago)
            query = query.filter(Run.start_time >= since)
        for run in query:  # Can not use query.delete with join
            session.delete(run)
        session.commit()

    def store_artifact(
            self, session, key, artifact, uid, iter=None, tag="", project=""):
        project = project or config.default_project
        self._create_project_if_not_exists(session, project)
        artifact = artifact.copy()
        updated = artifact.get("updated")
        if not updated:
            updated = artifact["updated"] = datetime.now(timezone.utc)
        if iter:
            key = "{}-{}".format(iter, key)
        art = self._get_artifact(session, uid, project, key)
        labels = artifact.get("labels", {})
        if not art:
            art = Artifact(
                key=key,
                uid=uid,
                updated=updated,
                project=project)
        update_labels(art, labels)
        art.struct = artifact
        self._upsert(session, art)
        if tag:
            self.tag_objects(session, [art], project, tag)

    def read_artifact(self, session, key, tag="", iter=None, project=""):
        project = project or config.default_project
        uid = self._resolve_tag(session, Artifact, project, tag)
        if iter:
            key = "{}-{}".format(iter, key)

        query = self._query(
            session, Artifact, key=key, project=project)
        if uid:
            query = query.filter(Artifact.uid == uid)
        else:
            # Select by last updated
            max_updated = session.query(
                func.max(Artifact.updated)).filter(
                Artifact.project == project, Artifact.key == key)
            query = query.filter(Artifact.updated.in_(max_updated))

        art = query.one_or_none()
        if not art:
            raise DBError(f"Artifact {key}:{tag}:{project} not found")
        return art.struct

    def list_artifacts(
            self, session, name=None, project=None, tag=None, labels=None,
            since=None, until=None):
        project = project or config.default_project
        uid = "latest"
        if tag:
            uid = self._resolve_tag(session, Artifact, project, tag)

        arts = ArtifactList(
            obj.struct
            for obj in self._find_artifacts(session, project, uid, labels, since, until)
        )
        return arts

    def del_artifact(self, session, key, tag="", project=""):
        project = project or config.default_project
        kw = {
            "key": key,
            "project": project,
        }
        if tag:
            kw["tag"] = tag

        self._delete(session, Artifact, **kw)

    def del_artifacts(
            self, session, name="", project="", tag="", labels=None):
        project = project or config.default_project
        for obj in self._find_artifacts(session, project, tag, labels, None, None):
            session.delete(obj)
        session.commit()

    def store_function(self, session, func, name, project="", tag=""):
        project = project or config.default_project
        self._create_project_if_not_exists(session, project)
        tag = tag or get_in(func, "metadata.tag") or "latest"

        # uid = self._resolve_tag(Function, project, tag)
        updated = datetime.now(timezone.utc)
        update_in(func, "metadata.updated", updated)
        #        fn = self._get_function(name, project, uid)
        fn = self._get_function(session, name, project, tag)
        if not fn:
            fn = Function(
                name=name,
                project=project,
                uid=tag,
            )
        fn.updated = updated
        labels = get_in(func, "metadata.labels", {})
        update_labels(fn, labels)
        fn.struct = func
        self._upsert(session, fn)

    def get_function(self, session, name, project="", tag=""):
        project = project or config.default_project
        query = self._query(session, Function, name=name, project=project)
        tag = tag or "latest"
        # if tag:
        #     if tag == "latest":
        #         uid = self._function_latest_uid(session, project, name)
        #         if not uid:
        #             raise DBError(
        #                 f"no latest version for function {project}:{name}")
        #     else:
        #         uid = self._resolve_tag(Function, project, tag)
        query = query.filter(Function.uid == tag)
        obj = query.one_or_none()
        if obj:
            return obj.struct

    def list_functions(
            self, session, name, project=None, tag=None, labels=None):
        project = project or config.default_project
        uid = tag
        # if uid:
        #     uid = self._resolve_tag(Function, project, uid)

        funcs = FunctionList()
        funcs.extend(
            obj.struct
            for obj in self._find_functions(session, name, project, uid, labels)
        )
        return funcs

    def list_artifact_tags(self, session, project):
        query = session.query(Artifact.Tag.name).filter(
            Artifact.Tag.project == project).distinct()
        return [row[0] for row in query]

    def store_schedule(self, session, data):
        sched = Schedule()
        sched.struct = data
        self._upsert(session, sched)

    def list_schedules(self, session):
        return [s.struct for s in session.query(Schedule)]

    def tag_objects(self, session, objs, project: str, name: str):
        """Tag objects with (project, name) tag.

        If force==True will update tag
        """
        for obj in objs:
            tag = obj.Tag(project=project, name=name, obj_id=obj.id)
            session.add(tag)
        session.commit()

    def del_tag(self, session, project: str, name: str):
        """Remove tag (project, name) from all objects"""
        count = 0
        for cls in _tagged:
            for obj in self._query(session, cls.Tag, project=project, name=name):
                session.delete(obj)
                count += 1
        session.commit()
        return count

    def find_tagged(self, session, project: str, name: str):
        """Return all objects tagged with (project, name)

        If not tag found, will return an empty str.
        """
        db_objects = []
        for cls in _tagged:
            for tag in self._query(session, cls.Tag, project=project, name=name):
                db_objects.append(self._query(session, cls).get(tag.obj_id))

        # TODO: this shouldn't return the db objects as is, sometimes they might be encoded with pickle, should
        #  something like:
        # objects = [db_object.struct if hasattr(db_object, "struct") else db_object for db_object in db_objects]
        return db_objects

    def list_tags(self, session, project: str):
        """Return all tags for a project"""
        tags = set()
        for cls in _tagged:
            for tag in self._query(session, cls.Tag, project=project):
                tags.add(tag.name)
        return tags

    def add_project(self, session, project: dict):
        project = project.copy()
        name = project.get("name")
        if not name:
            raise ValueError("project missing name")

        user_names = project.pop("users", [])
        prj = Project(**project)
        users = []  # self._find_or_create_users(session, user_names)
        prj.users.extend(users)
        self._upsert(session, prj)
        self._projects.add(prj.name)
        return prj.id

    def update_project(self, session, name, data: dict):
        prj = self.get_project(session, name)
        if not prj:
            raise DBError(f"unknown project - {name}")

        data = data.copy()
        user_names = data.pop("users", [])
        for key, value in data.items():
            if not hasattr(prj, key):
                raise DBError(f"unknown project attribute - {key}")
            setattr(prj, key, value)

        users = []  # self._find_or_create_users(session, user_names)
        prj.users.clear()
        prj.users.extend(users)
        self._upsert(session, prj, ignore=True)

    def get_project(self, session, name=None, project_id=None):
        if not (project_id or name):
            raise ValueError("need at least one of project_id or name")

        if project_id:
            return self._query(session, Project).get(project_id)

        return self._query(session, Project, name=name).one_or_none()

    def list_projects(self, session, owner=None):
        return self._query(session, Project, owner=owner)

    def _resolve_tag(self, session, cls, project, name):
        for tag in self._query(session, cls.Tag, project=project, name=name):
            return self._query(session, cls).get(tag.obj_id).uid
        return name  # Not found, return original uid

    def _query(self, session, cls, **kw):
        kw = {k: v for k, v in kw.items() if v is not None}
        return session.query(cls).filter_by(**kw)

    def _function_latest_uid(self, session, project, name):
        # FIXME
        query = (
            self._query(session, Function.uid).filter(Function.project == project, Function.name == name)
                .order_by(Function.updated.desc())
        ).limit(1)
        out = query.one_or_none()
        if out:
            return out[0]

    def _create_project_if_not_exists(self, session, name):
        if name not in self._projects:

            # fill cache from DB
            projects = self.list_projects(session)
            for project in projects:
                self._projects.add(project.name)
            if name not in self._projects:
                self.add_project(session, {"name": name})

    def _find_or_create_users(self, session, user_names):
        users = list(self._query(session, User).filter(User.name.in_(user_names)))
        new = set(user_names) - {user.name for user in users}
        if new:
            for name in new:
                user = User(name=name)
                session.add(user)
                users.append(user)
            try:
                session.commit()
            except SQLAlchemyError as err:
                session.rollback()
                raise DBError(f"add user: {err}") from err
        return users

    def _get_function(self, session, name, project, tag):
        uid = self._resolve_tag(session, Function, project, tag)
        query = self._query(session, Function, name=name, project=project, uid=uid)
        return query.one_or_none()

    def _get_artifact(self, session, uid, project, key):
        try:
            resp = self._query(
                session, Artifact, uid=uid, project=project, key=key).one_or_none()
            return resp
        finally:
            pass

    def _get_run(self, session, uid, project, iteration):
        try:
            resp = self._query(
                session, Run, uid=uid, project=project, iteration=iteration).one_or_none()
            return resp
        finally:
            pass

    def _delete_empty_labels(self, session, cls):
        session.query(cls).filter(cls.parent == NULL).delete()
        session.commit()

    def _upsert(self, session, obj, ignore=False):
        try:
            session.add(obj)
            session.commit()
        except SQLAlchemyError as err:
            session.rollback()
            cls = obj.__class__.__name__
            logger.warning(f"conflict adding {cls}, {err}")
            if not ignore:
                raise DBError(f"duplicate {cls} - {err}") from err

    def _find_runs(self, session, uid, project, labels, state):
        labels = label_set(labels)
        query = self._query(session, Run, uid=uid, project=project, state=state)
        return self._add_labels_filter(session, query, Run, labels)

    def _latest_uid_filter(self, session, query):
        # Create a sub query of latest uid (by updated) per (project,key)
        subq = session.query(
            Artifact.uid,
            Artifact.project,
            Artifact.key,
            func.max(Artifact.updated),
        ).group_by(
            Artifact.project,
            Artifact.key.label("key"),
        ).subquery("max_key")

        # Join curreny query with sub query on (project, key, uid)
        return query.join(
            subq,
            and_(
                Artifact.project == subq.c.project,
                Artifact.key == subq.c.key,
                Artifact.uid == subq.c.uid,
            )
        )

    def _find_artifacts(self, session, project, uid, labels, since, until):
        labels = label_set(labels)
        query = self._query(session, Artifact, project=project)
        if uid != "*":
            if uid == "latest":
                query = self._latest_uid_filter(session, query)
            else:
                query = query.filter(Artifact.uid == uid)

        query = self._add_labels_filter(session, query, Artifact, labels)

        if since or until:
            since = since or datetime.min
            until = until or datetime.max
            query = query.filter(and_(
                Artifact.updated >= since,
                Artifact.updated <= until
            ))

        return query

    def _find_functions(self, session, name, project, uid, labels):
        query = self._query(session, Function, name=name, project=project)
        if uid:
            query = query.filter(Function.uid == uid)

        labels = label_set(labels)
        return self._add_labels_filter(session, query, Function, labels)

    def _delete(self, session, cls, **kw):
        query = session.query(cls).filter_by(**kw)
        for obj in query:
            session.delete(obj)
        session.commit()

    def _find_lables(self, session, cls, label_cls, labels):
        return session.query(cls).join(label_cls).filter(
            label_cls.name.in_(labels))

    def _add_labels_filter(self, session, query, cls, labels):
        if not labels:
            return query

        preds = []
        for lbl in labels:
            if "=" in lbl:
                name, value = [v.strip() for v in lbl.split("=", 1)]
                cond = and_(cls.Label.name == name, cls.Label.value == value)
                preds.append(cond)
            else:
                preds.append(cls.Label.name == lbl.strip())

        subq = session.query(cls.Label).filter(*preds).subquery("labels")
        return query.join(subq)
