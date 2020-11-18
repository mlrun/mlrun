from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, List, Dict

import pytz
from sqlalchemy import and_, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import mlrun.errors
from mlrun.api import schemas
from mlrun.api.db.base import DBError, DBInterface
from mlrun.api.db.sqldb.helpers import (
    label_set,
    run_start_time,
    run_labels,
    run_state,
    update_labels,
)
from mlrun.api.db.sqldb.models import (
    Artifact,
    Function,
    Log,
    Run,
    Schedule,
    User,
    Project,
    FeatureSet,
    Feature,
    Entity,
    _tagged,
    _labeled,
)
from mlrun.config import config
from mlrun.lists import ArtifactList, FunctionList, RunList
from mlrun.utils import (
    get_in,
    update_in,
    logger,
    fill_function_hash,
    fill_object_hash,
    generate_object_uri,
    match_times,
)
import mergedeep

NULL = None  # Avoid flake8 issuing warnings when comparing in filter
run_time_fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
unversioned_tagged_object_uid_prefix = "unversioned-"


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
        self._ensure_project(session, project)
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

    def delete_log(self, session: Session, project: str, uid: str):
        project = project or config.default_project
        self._delete(session, Log, project=project, uid=uid)

    def _delete_logs(self, session: Session, project: str):
        logger.debug("Removing logs from db", project=project)
        for log in self._query(session, Log, project=project):
            self.delete_log(session, project, log.uid)

    def store_run(self, session, run_data, uid, project="", iter=0):
        project = project or config.default_project
        logger.debug(
            "Storing run to db", project=project, uid=uid, iter=iter, run=run_data
        )
        self._ensure_project(session, project)
        run = self._get_run(session, uid, project, iter)
        if not run:
            run = Run(
                uid=uid,
                project=project,
                iteration=iter,
                state=run_state(run_data),
                start_time=run_start_time(run_data) or datetime.now(timezone.utc),
            )
        labels = run_labels(run_data)
        new_state = run_state(run_data)
        if new_state:
            run.state = new_state
        update_labels(run, labels)
        run.struct = run_data
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

    def read_run(self, session, uid, project=None, iter=0):
        project = project or config.default_project
        run = self._get_run(session, uid, project, iter)
        if not run:
            raise mlrun.errors.MLRunNotFoundError(f"Run {uid}:{project} not found")
        return run.struct

    def list_runs(
        self,
        session,
        name=None,
        uid=None,
        project=None,
        labels=None,
        state=None,
        sort=True,
        last=0,
        iter=False,
        start_time_from=None,
        start_time_to=None,
        last_update_time_from=None,
        last_update_time_to=None,
    ):
        project = project or config.default_project
        query = self._find_runs(session, uid, project, labels)
        if start_time_from:
            query = query.filter(Run.start_time >= start_time_from)
        if start_time_to:
            query = query.filter(Run.start_time <= start_time_to)
        if sort:
            query = query.order_by(Run.start_time.desc())
        if last:
            query = query.limit(last)
        if not iter:
            query = query.filter(Run.iteration == 0)

        filtered_runs = self._post_query_runs_filter(
            query, name, state, last_update_time_from, last_update_time_to
        )

        runs = RunList()
        for run in filtered_runs:
            runs.append(run.struct)

        return runs

    def del_run(self, session, uid, project=None, iter=0):
        project = project or config.default_project
        # We currently delete *all* iterations
        self._delete(session, Run, uid=uid, project=project)

    def del_runs(
        self, session, name=None, project=None, labels=None, state=None, days_ago=0
    ):
        # FIXME: Run has no `name`
        project = project or config.default_project
        query = self._find_runs(session, None, project, labels)
        if days_ago:
            since = datetime.now(timezone.utc) - timedelta(days=days_ago)
            query = query.filter(Run.start_time >= since)
        filtered_runs = self._post_query_runs_filter(query, name, state)
        for run in filtered_runs:  # Can not use query.delete with join
            session.delete(run)
        session.commit()

    def store_artifact(
        self, session, key, artifact, uid, iter=None, tag="", project=""
    ):
        project = project or config.default_project
        self._ensure_project(session, project)
        artifact = artifact.copy()
        updated = artifact.get("updated")
        if not updated:
            updated = artifact["updated"] = datetime.now(timezone.utc)
        if iter:
            key = "{}-{}".format(iter, key)
        art = self._get_artifact(session, uid, project, key)
        labels = artifact.get("labels", {})
        if not art:
            art = Artifact(key=key, uid=uid, updated=updated, project=project)
        update_labels(art, labels)
        art.struct = artifact
        self._upsert(session, art)
        tag = tag or "latest"
        self.tag_objects(session, [art], project, tag)

    def read_artifact(self, session, key, tag="", iter=None, project=""):
        project = project or config.default_project
        uids = self._resolve_tag(session, Artifact, project, tag)
        if iter:
            key = "{}-{}".format(iter, key)

        query = self._query(session, Artifact, key=key, project=project)

        # TODO: refactor this
        # tag has 2 meanings:
        # 1. tag - in this case _resolve_tag will find the relevant uids and will return a list
        # 2. uid - in this case _resolve_tag won't find anything and simply return what was given to it, which actually
        # represents the uid
        if isinstance(uids, list) and uids:
            query = query.filter(Artifact.uid.in_(uids))
        elif isinstance(uids, str) and uids:
            query = query.filter(Artifact.uid == uids)
        else:
            # Select by last updated
            max_updated = session.query(func.max(Artifact.updated)).filter(
                Artifact.project == project, Artifact.key == key
            )
            query = query.filter(Artifact.updated.in_(max_updated))

        art = query.one_or_none()
        if not art:
            raise DBError(f"Artifact {key}:{tag}:{project} not found")
        return art.struct

    def list_artifacts(
        self,
        session,
        name=None,
        project=None,
        tag=None,
        labels=None,
        since=None,
        until=None,
        kind=None,
        category: schemas.ArtifactCategories = None,
    ):
        project = project or config.default_project

        # TODO: Refactor this area
        # in case where tag is not given uids will be "latest" to mark to _find_artifacts to find the latest using the
        # old way - by the updated field
        uids = "latest"
        if tag:
            uids = self._resolve_tag(session, Artifact, project, tag)

        artifacts = ArtifactList(
            artifact.struct
            for artifact in self._find_artifacts(
                session, project, uids, labels, since, until, name, kind, category
            )
        )
        return artifacts

    def del_artifact(self, session, key, tag="", project=""):
        project = project or config.default_project
        self._delete_artifact_tags(session, project, key, tag, commit=False)
        self._delete_class_labels(
            session, Artifact, project=project, key=key, commit=False
        )
        kw = {
            "key": key,
            "project": project,
        }
        if tag:
            kw["tag"] = tag

        self._delete(session, Artifact, **kw)

    def _delete_artifact_tags(
        self, session, project, artifact_key, tag_name="", commit=True
    ):
        query = (
            session.query(Artifact.Tag)
            .join(Artifact)
            .filter(Artifact.project == project, Artifact.key == artifact_key)
        )
        if tag_name:
            query = query.filter(Artifact.Tag.name == tag_name)
        for tag in query:
            session.delete(tag)
        if commit:
            session.commit()

    def del_artifacts(self, session, name="", project="", tag="*", labels=None):
        project = project or config.default_project
        for artifact in self._find_artifacts(session, project, tag, labels, name=name):
            self.del_artifact(session, artifact.key, "", project)

    def store_function(
        self, session, function, name, project="", tag="", versioned=False
    ):
        project = project or config.default_project
        self._ensure_project(session, project)
        tag = tag or get_in(function, "metadata.tag") or "latest"
        hash_key = fill_function_hash(function, tag)

        # clear tag from object in case another function will "take" that tag
        update_in(function, "metadata.tag", "")

        # versioned means whether we want to version this function object so that it will queryable by its hash key
        # to enable that we set the uid to the hash key so it will have a unique record (Unique constraint of function
        # is the set (project, name, uid))
        # when it's not enabled it means we want to have one unique function object for the set (project, name, tag)
        # that will be reused on every store function (cause we don't want to version each version e.g. create a new
        # record) so we set the uid to be unversioned-{tag}
        if versioned:
            uid = hash_key
        else:
            uid = f"{unversioned_tagged_object_uid_prefix}{tag}"

        updated = datetime.now(timezone.utc)
        update_in(function, "metadata.updated", updated)
        fn = self._get_function(session, name, project, uid)
        if not fn:
            fn = Function(name=name, project=project, uid=uid,)
        fn.updated = updated
        labels = get_in(function, "metadata.labels", {})
        update_labels(fn, labels)
        fn.struct = function
        self._upsert(session, fn)
        self.tag_objects_v2(session, [fn], project, tag)
        return hash_key

    def get_function(self, session, name, project="", tag="", hash_key=""):
        project = project or config.default_project
        query = self._query(session, Function, name=name, project=project)
        computed_tag = tag or "latest"
        tag_function_uid = None
        if not tag and hash_key:
            uid = hash_key
        else:
            tag_function_uid = self._resolve_class_tag_uid(
                session, Function, project, name, computed_tag
            )
            if tag_function_uid is None:
                function_uri = generate_object_uri(project, name, tag)
                raise mlrun.errors.MLRunNotFoundError(
                    f"Function tag not found {function_uri}"
                )
            uid = tag_function_uid
        if uid:
            query = query.filter(Function.uid == uid)
        obj = query.one_or_none()
        if obj:
            function = obj.struct

            # If queried by hash key remove status
            if hash_key:
                function["status"] = None

            # If connected to a tag add it to metadata
            if tag_function_uid:
                function["metadata"]["tag"] = computed_tag
            return function
        else:
            function_uri = generate_object_uri(project, name, tag, hash_key)
            raise mlrun.errors.MLRunNotFoundError(f"Function not found {function_uri}")

    def delete_function(self, session: Session, project: str, name: str):
        logger.debug("Removing function from db", project=project, name=name)
        self._delete_function_tags(session, project, name, commit=False)
        self._delete_class_labels(
            session, Function, project=project, name=name, commit=False
        )
        self._delete(session, Function, project=project, name=name)

    def _delete_functions(self, session: Session, project: str):
        for function in self._query(session, Function, project=project):
            self.delete_function(session, project, function.name)

    def _delete_resources_tags(self, session: Session, project: str):
        for tagged_class in _tagged:
            self._delete(session, tagged_class, project=project)

    def _delete_resources_labels(self, session: Session, project: str):
        for labeled_class in _labeled:
            if hasattr(labeled_class, "project"):
                self._delete(session, labeled_class, project=project)

    def list_functions(self, session, name=None, project=None, tag=None, labels=None):
        project = project or config.default_project
        uids = None
        if tag:
            uids = self._resolve_class_tag_uids(session, Function, project, tag, name)
        functions = FunctionList()
        for function in self._find_functions(session, name, project, uids, labels):
            function_dict = function.struct
            if not tag:
                function_tags = self._list_function_tags(session, project, function.id)
                if len(function_tags) == 0:

                    # function status should be added only to tagged functions
                    function_dict["status"] = None

                    # the unversioned uid is only a place holder for tagged instance that are is versioned
                    # if another instance "took" the tag, we're left with an unversioned untagged instance
                    # don't list it
                    if function.uid.startswith(unversioned_tagged_object_uid_prefix):
                        continue

                    functions.append(function_dict)
                elif len(function_tags) == 1:
                    function_dict["metadata"]["tag"] = function_tags[0]
                    functions.append(function_dict)
                else:
                    for function_tag in function_tags:
                        function_dict_copy = deepcopy(function_dict)
                        function_dict_copy["metadata"]["tag"] = function_tag
                        functions.append(function_dict_copy)
            else:
                function_dict["metadata"]["tag"] = tag
                functions.append(function_dict)
        return functions

    def _delete_function_tags(self, session, project, function_name, commit=True):
        query = session.query(Function.Tag).filter(
            Function.Tag.project == project, Function.Tag.obj_name == function_name
        )
        for obj in query:
            session.delete(obj)
        if commit:
            session.commit()

    def _list_function_tags(self, session, project, function_id):
        query = (
            session.query(Function.Tag.name)
            .filter(Function.Tag.project == project, Function.Tag.obj_id == function_id)
            .distinct()
        )
        return [row[0] for row in query]

    def list_artifact_tags(self, session, project):
        query = (
            session.query(Artifact.Tag.name)
            .filter(Artifact.Tag.project == project)
            .distinct()
        )
        return [row[0] for row in query]

    def create_schedule(
        self,
        session: Session,
        project: str,
        name: str,
        kind: schemas.ScheduleKinds,
        scheduled_object: Any,
        cron_trigger: schemas.ScheduleCronTrigger,
        labels: Dict = None,
    ):
        self._ensure_project(session, project)
        schedule = Schedule(
            project=project,
            name=name,
            kind=kind.value,
            creation_time=datetime.now(timezone.utc),
            # these are properties of the object that map manually (using getters and setters) to other column of the
            # table and therefore Pycharm yells that they're unexpected
            scheduled_object=scheduled_object,
            cron_trigger=cron_trigger,
        )

        labels = labels or {}
        update_labels(schedule, labels)

        logger.debug(
            "Saving schedule to db",
            project=project,
            name=name,
            kind=kind,
            cron_trigger=cron_trigger,
        )
        self._upsert(session, schedule)

    def update_schedule(
        self,
        session: Session,
        project: str,
        name: str,
        scheduled_object: Any = None,
        cron_trigger: schemas.ScheduleCronTrigger = None,
        labels: Dict = None,
        last_run_uri: str = None,
    ):
        self._ensure_project(session, project)
        query = self._query(session, Schedule, project=project, name=name)
        schedule = query.one_or_none()

        # explicitly ensure the updated fields are not None, as they can be empty strings/dictionaries etc.
        if scheduled_object is not None:
            schedule.scheduled_object = scheduled_object

        if cron_trigger is not None:
            schedule.cron_trigger = cron_trigger

        if labels is not None:
            update_labels(schedule, labels)

        if last_run_uri is not None:
            schedule.last_run_uri = last_run_uri

        logger.debug(
            "Updating schedule in db",
            project=project,
            name=name,
            cron_trigger=cron_trigger,
            labels=labels,
        )
        session.merge(schedule)
        session.commit()

    def list_schedules(
        self,
        session: Session,
        project: str = None,
        name: str = None,
        labels: str = None,
        kind: schemas.ScheduleKinds = None,
    ) -> List[schemas.ScheduleRecord]:
        logger.debug("Getting schedules from db", project=project, name=name, kind=kind)
        query = self._query(session, Schedule, project=project, kind=kind)
        if name is not None:
            query = query.filter(Schedule.name.ilike(f"%{name}%"))
        labels = label_set(labels)
        query = self._add_labels_filter(session, query, Schedule, labels)

        schedules = [
            self._transform_schedule_model_to_scheme(db_schedule)
            for db_schedule in query
        ]
        return schedules

    def get_schedule(
        self, session: Session, project: str, name: str
    ) -> schemas.ScheduleRecord:
        logger.debug("Getting schedule from db", project=project, name=name)
        query = self._query(session, Schedule, project=project, name=name)
        db_schedule = query.one_or_none()
        schedule = self._transform_schedule_model_to_scheme(db_schedule)
        return schedule

    def delete_schedule(self, session: Session, project: str, name: str):
        logger.debug("Removing schedule from db", project=project, name=name)
        self._delete_class_labels(
            session, Schedule, project=project, name=name, commit=False
        )
        self._delete(session, Schedule, project=project, name=name)

    def _delete_schedules(self, session: Session, project: str):
        logger.debug("Removing schedules from db", project=project)
        for schedule in self.list_schedules(session, project=project):
            self.delete_schedule(session, project, schedule.name)

    def _delete_feature_sets(self, session: Session, project: str):
        logger.debug("Removing feature-sets from db", project=project)
        for feature_set in self.list_feature_sets(session, project).feature_sets:
            self.delete_feature_set(session, project, feature_set.metadata.name)

    def tag_objects(self, session, objs, project: str, name: str):
        """Tag objects with (project, name) tag.

        If force==True will update tag
        """
        for obj in objs:
            tag = obj.Tag(project=project, name=name, obj_id=obj.id)
            self._upsert(session, tag, ignore=True)

    def tag_objects_v2(self, session, objs, project: str, name: str):
        """Tag objects with (project, name) tag.

        If force==True will update tag
        """
        for obj in objs:
            query = self._query(
                session, obj.Tag, name=name, project=project, obj_name=obj.name
            )
            tag = query.one_or_none()
            if not tag:
                tag = obj.Tag(project=project, name=name, obj_name=obj.name)
            tag.obj_id = obj.id
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

        project.pop("users", [])
        prj = Project(**project)
        users = (
            []
        )  # self._find_or_create_users(session, user_names) - user_names are previously popped users
        prj.users.extend(users)
        self._upsert(session, prj)
        self._projects.add(prj.name)
        return prj.id

    def update_project(self, session, name, data: dict):
        prj = self.get_project(session, name)
        if not prj:
            raise DBError(f"unknown project - {name}")

        data = data.copy()
        data.pop("users", [])
        for key, value in data.items():
            if not hasattr(prj, key):
                raise DBError(f"unknown project attribute - {key}")
            setattr(prj, key, value)

        users = (
            []
        )  # self._find_or_create_users(session, user_names) - user_names are previously popped users
        prj.users.clear()
        prj.users.extend(users)
        self._upsert(session, prj, ignore=True)

    def get_project(self, session, name=None, project_id=None):
        if not (project_id or name):
            raise ValueError("need at least one of project_id or name")

        if project_id:
            return self._query(session, Project).get(project_id)

        return self._query(session, Project, name=name).one_or_none()

    def delete_project(self, session, name: str):
        self.del_artifacts(session, project=name)
        self._delete_logs(session, name)
        self.del_runs(session, project=name)
        self._delete_schedules(session, name)
        self._delete_functions(session, name)
        self._delete_feature_sets(session, name)

        # resources deletion should remove their tags and labels as well, but doing another try in case there are
        # orphan resources
        self._delete_resources_tags(session, name)
        self._delete_resources_labels(session, name)
        self._delete(session, Project, name=name)

    def list_projects(self, session, owner=None):
        return self._query(session, Project, owner=owner)

    def _get_feature_set(
        self, session, project: str, name: str, tag: str = None, uid: str = None,
    ):
        query = self._query(session, FeatureSet, name=name, project=project)
        computed_tag = tag or "latest"
        feature_set_tag_uid = None
        if tag or not uid:
            feature_set_tag_uid = self._resolve_class_tag_uid(
                session, FeatureSet, project, name, computed_tag
            )
            if feature_set_tag_uid is None:
                return None
            uid = feature_set_tag_uid
        if uid:
            query = query.filter(FeatureSet.uid == uid)
        db_feature_set = query.one_or_none()
        if db_feature_set:
            feature_set = self._transform_feature_set_model_to_schema(db_feature_set)

            # If connected to a tag add it to metadata
            if feature_set_tag_uid:
                feature_set.metadata.tag = computed_tag
            return feature_set
        else:
            return None

    def get_feature_set(
        self, session, project: str, name: str, tag: str = None, uid: str = None,
    ) -> schemas.FeatureSet:
        feature_set = self._get_feature_set(session, project, name, tag, uid)
        if not feature_set:
            feature_set_uri = generate_object_uri(project, name, tag)
            raise mlrun.errors.MLRunNotFoundError(
                f"Feature-set not found {feature_set_uri}"
            )

        return feature_set

    def _get_feature_sets_tags_map(self, session, project, tag, name=None):
        # Find object IDs by tag, project and feature-set-name (which is a like query)
        tag_query = self._query(session, FeatureSet.Tag, project=project, name=tag)
        if name:
            tag_query = tag_query.filter(FeatureSet.Tag.obj_name.ilike(f"%{name}%"))

        # Generate a mapping from each object id (note: not uid, it's the DB ID) to its associated tags.
        obj_id_tags = {}
        for row in tag_query:
            if row.obj_id in obj_id_tags:
                obj_id_tags[row.obj_id].append(row.name)
            else:
                obj_id_tags[row.obj_id] = [row.name]
        return obj_id_tags

    def _generate_feature_set_results_with_tags_assigned(
        self, feature_set_record, obj_id_tags, default_tag=None
    ):
        # Using a similar mechanism here to assign tags to feature sets as is used in list_functions. Please refer
        # there for some comments explaining the logic.
        feature_sets = []
        if default_tag:
            feature_sets.append(
                self._transform_feature_set_model_to_schema(
                    feature_set_record, default_tag
                )
            )
        else:
            feature_set_tags = obj_id_tags.get(feature_set_record.id, [])
            if len(feature_set_tags) == 0 and not feature_set_record.uid.startswith(
                unversioned_tagged_object_uid_prefix
            ):
                feature_set = self._transform_feature_set_model_to_schema(
                    feature_set_record
                )
                feature_sets.append(feature_set)
            else:
                for feature_set_tag in feature_set_tags:
                    feature_sets.append(
                        self._transform_feature_set_model_to_schema(
                            feature_set_record, feature_set_tag
                        )
                    )
        return feature_sets

    @staticmethod
    def _generate_feature_set_digest(feature_set: schemas.FeatureSet):
        return schemas.FeatureSetDigestOutput(
            metadata=feature_set.metadata,
            spec=schemas.FeatureSetDigestSpec(entities=feature_set.spec.entities),
        )

    def list_features(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        entities: List[str] = None,
        labels: List[str] = None,
    ) -> schemas.FeaturesOutput:
        # We don't filter by feature-set name here, as the name parameter refers to features
        feature_set_id_tags = self._get_feature_sets_tags_map(
            session, project, tag, name=None
        )

        # Query the actual objects to be returned
        query = (
            session.query(FeatureSet, Feature)
            .filter_by(project=project)
            .join(FeatureSet.features)
        )

        if name:
            query = query.filter(Feature.name.ilike(f"%{name}%"))
        if labels:
            query = self._add_labels_filter(session, query, Feature, labels)
        if tag:
            query = query.filter(FeatureSet.id.in_(feature_set_id_tags.keys()))
        if entities:
            query = query.join(FeatureSet.entities).filter(Entity.name.in_(entities))

        features_results = []
        for row in query:
            feature_record = schemas.FeatureRecord.from_orm(row.Feature)
            feature_name = feature_record.name

            feature_sets = self._generate_feature_set_results_with_tags_assigned(
                row.FeatureSet, feature_set_id_tags, tag
            )

            for feature_set in feature_sets:
                # Get the feature from the feature-set full structure, as it may contain extra fields (which are not
                # in the DB)
                feature = next(
                    feature
                    for feature in feature_set.spec.features
                    if feature.name == feature_name
                )

                features_results.append(
                    schemas.FeatureListOutput(
                        feature=feature,
                        feature_set_digest=self._generate_feature_set_digest(
                            feature_set
                        ),
                    )
                )
        return schemas.FeaturesOutput(features=features_results)

    def list_feature_sets(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: List[str] = None,
        features: List[str] = None,
        labels: List[str] = None,
    ) -> schemas.FeatureSetsOutput:
        obj_id_tags = self._get_feature_sets_tags_map(session, project, tag, name)

        # Query the actual objects to be returned
        query = self._query(session, FeatureSet, project=project, state=state)

        if name is not None:
            query = query.filter(FeatureSet.name.ilike(f"%{name}%"))
        if tag:
            query = query.filter(FeatureSet.id.in_(obj_id_tags.keys()))
        if entities:
            query = query.join(FeatureSet.entities).filter(Entity.name.in_(entities))
        if features:
            query = query.join(FeatureSet.features).filter(Feature.name.in_(features))
        if labels:
            query = self._add_labels_filter(session, query, FeatureSet, labels)

        feature_sets = []
        for feature_set_record in query:
            feature_sets.extend(
                self._generate_feature_set_results_with_tags_assigned(
                    feature_set_record, obj_id_tags, tag
                )
            )
        return schemas.FeatureSetsOutput(feature_sets=feature_sets)

    @staticmethod
    def _update_feature_set_features(
        feature_set: FeatureSet, feature_dicts: List[dict], replace=False
    ):
        if replace:
            feature_set.features = []

        for feature_dict in feature_dicts:
            labels = feature_dict.get("labels") or {}
            feature = Feature(
                name=feature_dict["name"],
                value_type=feature_dict["value_type"],
                labels=[],
            )
            update_labels(feature, labels)
            feature_set.features.append(feature)

    @staticmethod
    def _update_feature_set_entities(
        feature_set: FeatureSet, entity_dicts: List[dict], replace=False
    ):
        if replace:
            feature_set.entities = []

        for entity_dict in entity_dicts:
            labels = entity_dict.get("labels") or {}
            entity = Entity(
                name=entity_dict["name"],
                value_type=entity_dict["value_type"],
                labels=[],
            )
            update_labels(entity, labels)
            feature_set.entities.append(entity)

    def _get_feature_set_instance_by_uid(self, session, name, project, uid):
        query = self._query(session, FeatureSet, name=name, project=project, uid=uid)
        return query.one_or_none()

    def _update_existing_feature_set(
        self, feature_set: FeatureSet, new_feature_set_dict: dict
    ):
        feature_set_spec = new_feature_set_dict.get("spec")
        if feature_set_spec:
            features = feature_set_spec.pop("features", [])
            entities = feature_set_spec.pop("entities", [])

        self._update_feature_set_features(feature_set, features)
        self._update_feature_set_entities(feature_set, entities)

        labels = new_feature_set_dict["metadata"].pop("labels", {})
        update_labels(feature_set, labels)

    def store_feature_set(
        self,
        session,
        project,
        name,
        feature_set: schemas.FeatureSet,
        tag=None,
        uid=None,
        versioned=True,
        always_overwrite=False,
    ):
        if not tag and not uid:
            raise ValueError("cannot store feature set without reference (tag or uid)")

        if feature_set.metadata.name != name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Changing name for an existing feature-set"
            )

        original_uid = uid

        existing_feature_set = self._get_feature_set(
            session, project, name, tag, original_uid
        )
        if not existing_feature_set:
            return self.create_feature_set(session, project, feature_set, versioned)

        feature_set_dict = feature_set.dict()
        uid = fill_object_hash(feature_set_dict, "uid", tag)
        if not versioned:
            uid = f"{unversioned_tagged_object_uid_prefix}{tag}"
            feature_set_dict["metadata"]["uid"] = uid

        # If object was referenced by UID, the request cannot modify it
        if original_uid and uid != original_uid:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Changing uid for an object referenced by its uid"
            )

        if uid == existing_feature_set.metadata.uid or always_overwrite:
            db_feature_set = self._get_feature_set_instance_by_uid(
                session, name, project, existing_feature_set.metadata.uid
            )
        else:
            db_feature_set = FeatureSet(project=project)

        db_feature_set.name = feature_set.metadata.name
        db_feature_set.updated = datetime.now(timezone.utc)
        db_feature_set.state = feature_set_dict.get("status", {}).get("state")
        db_feature_set.uid = uid
        db_feature_set.full_object = feature_set_dict

        self._update_existing_feature_set(db_feature_set, feature_set_dict)

        self._upsert(session, db_feature_set)
        self.tag_objects_v2(session, [db_feature_set], project, tag)

        return uid

    def create_feature_set(
        self, session, project, feature_set: schemas.FeatureSet, versioned=True
    ):
        name = feature_set.metadata.name
        if not name or not project:
            raise ValueError("feature-set missing name or project")

        self._ensure_project(session, project)
        tag = feature_set.metadata.tag or "latest"

        feature_set_dict = feature_set.dict()
        hash_key = fill_object_hash(feature_set_dict, "uid", tag)

        if versioned:
            uid = hash_key
        else:
            uid = f"{unversioned_tagged_object_uid_prefix}{tag}"
            feature_set_dict["metadata"]["uid"] = uid

        state = feature_set_dict.get("status", {}).get("state")

        feature_set = self._get_feature_set_instance_by_uid(session, name, project, uid)
        if feature_set:
            feature_set_uri = generate_object_uri(project, name, tag)
            raise mlrun.errors.MLRunConflictError(
                f"Adding an already-existing feature-set {feature_set_uri}"
            )
        else:
            feature_set = FeatureSet(
                name=name,
                project=project,
                state=state,
                uid=uid,
                full_object=feature_set_dict,
            )

        self._update_existing_feature_set(feature_set, feature_set_dict)

        self._upsert(session, feature_set)
        self.tag_objects_v2(session, [feature_set], project, tag)

        return uid

    def patch_feature_set(
        self,
        session,
        project,
        name,
        feature_set_update: dict,
        tag=None,
        uid=None,
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
    ):
        feature_set_record = self._get_feature_set(session, project, name, tag, uid)
        if not feature_set_record:
            feature_set_uri = generate_object_uri(project, name, tag)
            raise mlrun.errors.MLRunNotFoundError(
                f"Feature-set not found {feature_set_uri}"
            )

        feature_set_struct = feature_set_record.dict()
        # using mergedeep for merging the patch content into the existing dictionary
        strategy = mergedeep.Strategy.REPLACE
        if patch_mode.value == "additive":
            strategy = mergedeep.Strategy.ADDITIVE
        mergedeep.merge(feature_set_struct, feature_set_update, strategy=strategy)

        versioned = not feature_set_record.metadata.uid.startswith(
            unversioned_tagged_object_uid_prefix
        )
        feature_set = schemas.FeatureSet(**feature_set_struct)
        return self.store_feature_set(
            session,
            project,
            name,
            feature_set,
            feature_set.metadata.tag,
            uid,
            versioned,
            always_overwrite=True,
        )

    def delete_feature_set(self, session, project, name):
        self._delete(session, FeatureSet.Tag, project=project, obj_name=name)
        self._delete(session, FeatureSet, project=project, name=name)

    def _resolve_tag(self, session, cls, project, name):
        uids = []
        for tag in self._query(session, cls.Tag, project=project, name=name):
            uids.append(self._query(session, cls).get(tag.obj_id).uid)
        if not uids:
            return name  # Not found, return original uid
        return uids

    def _resolve_class_tag_uid(self, session, cls, project, obj_name, tag_name):
        for tag in self._query(
            session, cls.Tag, project=project, obj_name=obj_name, name=tag_name
        ):
            return self._query(session, cls).get(tag.obj_id).uid
        return None

    def _resolve_class_tag_uids(
        self, session, cls, project, tag_name, obj_name=None
    ) -> List[str]:
        uids = []

        query = self._query(session, cls.Tag, project=project, name=tag_name)
        if obj_name:
            query = query.filter(cls.Tag.obj_name.ilike(f"%{obj_name}%"))

        for tag in query:
            uids.append(self._query(session, cls).get(tag.obj_id).uid)
        return uids

    def _query(self, session, cls, **kw):
        kw = {k: v for k, v in kw.items() if v is not None}
        return session.query(cls).filter_by(**kw)

    def _function_latest_uid(self, session, project, name):
        # FIXME
        query = (
            self._query(session, Function.uid)
            .filter(Function.project == project, Function.name == name)
            .order_by(Function.updated.desc())
        ).limit(1)
        out = query.one_or_none()
        if out:
            return out[0]

    def _ensure_project(self, session, name):
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

    def _get_function(self, session, name, project, uid):
        query = self._query(session, Function, name=name, project=project, uid=uid)
        return query.one_or_none()

    def _get_artifact(self, session, uid, project, key):
        try:
            resp = self._query(
                session, Artifact, uid=uid, project=project, key=key
            ).one_or_none()
            return resp
        finally:
            pass

    def _get_run(self, session, uid, project, iteration):
        try:
            resp = self._query(
                session, Run, uid=uid, project=project, iteration=iteration
            ).one_or_none()
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

    def _find_runs(self, session, uid, project, labels):
        labels = label_set(labels)
        if project == "*":
            project = None
        query = self._query(session, Run, uid=uid, project=project)
        return self._add_labels_filter(session, query, Run, labels)

    def _post_query_runs_filter(
        self,
        query,
        name=None,
        state=None,
        last_update_time_from=None,
        last_update_time_to=None,
    ):
        """
        This function is hacky and exists to cover on bugs we had with how we save our data in the DB
        We're doing it the hacky way since:
        1. SQLDB is about to be replaced
        2. Schema + Data migration are complicated and as long we can avoid them, we prefer to (also because of 1)
        name - the name is only saved in the json itself, therefore we can't use the SQL query filter and have to filter
        it ourselves
        state - the state is saved in a column, but, there was a bug in which the state was only getting updated in the
        json itself, therefore, in field systems, most runs records will have an empty or not updated data in the state
        column
        """
        if (
            not name
            and not state
            and not last_update_time_from
            and not last_update_time_to
        ):
            return query.all()

        filtered_runs = []
        for run in query:
            run_json = run.struct
            if name:
                if (
                    not run_json
                    or not isinstance(run_json, dict)
                    or name not in run_json.get("metadata", {}).get("name")
                ):
                    continue
            if state:
                record_state = run.state
                json_state = None
                if (
                    run_json
                    and isinstance(run_json, dict)
                    and run_json.get("status", {}).get("state")
                ):
                    json_state = run_json.get("status", {}).get("state")
                if not record_state and not json_state:
                    continue
                # json_state has precedence over record state
                if json_state:
                    if state not in json_state:
                        continue
                else:
                    if state not in record_state:
                        continue
            if last_update_time_from or last_update_time_to:
                if not match_times(
                    last_update_time_from,
                    last_update_time_to,
                    run_json,
                    "status.last_update",
                ):
                    continue

            filtered_runs.append(run)

        return filtered_runs

    def _latest_uid_filter(self, session, query):
        # Create a sub query of latest uid (by updated) per (project,key)
        subq = (
            session.query(
                Artifact.uid,
                Artifact.project,
                Artifact.key,
                func.max(Artifact.updated),
            )
            .group_by(Artifact.project, Artifact.key.label("key"),)
            .subquery("max_key")
        )

        # Join curreny query with sub query on (project, key, uid)
        return query.join(
            subq,
            and_(
                Artifact.project == subq.c.project,
                Artifact.key == subq.c.key,
                Artifact.uid == subq.c.uid,
            ),
        )

    def _find_artifacts(
        self,
        session,
        project,
        uids,
        labels=None,
        since=None,
        until=None,
        name=None,
        kind=None,
        category: schemas.ArtifactCategories = None,
    ):
        """
        TODO: refactor this method
        basically uids should be list of strings (representing uids), but we also handle two special cases (mainly for
        BC until we refactor the whole artifacts API):
        1. uids == "*" - in which we don't care about uids we just don't add any filter for this column
        1. uids == "latest" - in which we find the relevant uid by finding the latest artifact using the updated column
        """
        if category and kind:
            message = "Category and Kind filters can't be given together"
            logger.warning(message, kind=kind, category=category)
            raise ValueError(message)
        labels = label_set(labels)
        query = self._query(session, Artifact, project=project)
        if uids != "*":
            if uids == "latest":
                query = self._latest_uid_filter(session, query)
            else:
                query = query.filter(Artifact.uid.in_(uids))

        query = self._add_labels_filter(session, query, Artifact, labels)

        if since or until:
            since = since or datetime.min
            until = until or datetime.max
            query = query.filter(
                and_(Artifact.updated >= since, Artifact.updated <= until)
            )

        if name is not None:
            query = query.filter(Artifact.key.ilike(f"%{name}%"))

        if kind:
            return self._filter_artifacts_by_kinds(query, [kind])

        elif category:
            return self._filter_artifacts_by_category(query, category)

        else:
            return query.all()

    def _filter_artifacts_by_category(
        self, artifacts, category: schemas.ArtifactCategories
    ):
        kinds, exclude = category.to_kinds_filter()
        return self._filter_artifacts_by_kinds(artifacts, kinds, exclude)

    def _filter_artifacts_by_kinds(
        self, artifacts, kinds: List[str], exclude: bool = False
    ):
        """
        :param kinds - list of kinds to filter by
        :param exclude - if true then the filter will be "all except" - get all artifacts excluding the ones who have
         any of the given kinds
        """
        # see docstring of _post_query_runs_filter for why we're filtering it manually
        filtered_artifacts = []
        for artifact in artifacts:
            artifact_json = artifact.struct
            if (
                artifact_json
                and isinstance(artifact_json, dict)
                and (
                    (
                        not exclude
                        and any([kind == artifact_json.get("kind") for kind in kinds])
                    )
                    or (
                        exclude
                        and all([kind != artifact_json.get("kind") for kind in kinds])
                    )
                )
            ):
                filtered_artifacts.append(artifact)
        return filtered_artifacts

    def _find_functions(self, session, name, project, uids=None, labels=None):
        query = self._query(session, Function, name=name, project=project)
        if uids:
            query = query.filter(Function.uid.in_(uids))

        labels = label_set(labels)
        return self._add_labels_filter(session, query, Function, labels)

    def _delete(self, session, cls, **kw):
        query = session.query(cls).filter_by(**kw)
        for obj in query:
            session.delete(obj)
        session.commit()

    def _find_lables(self, session, cls, label_cls, labels):
        return session.query(cls).join(label_cls).filter(label_cls.name.in_(labels))

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

    def _delete_class_labels(
        self,
        session: Session,
        cls: Any,
        project: str = "",
        name: str = "",
        key: str = "",
        commit: bool = True,
    ):
        filters = []
        if project:
            filters.append(cls.project == project)
        if name:
            filters.append(cls.name == name)
        if key:
            filters.append(cls.key == key)
        query = session.query(cls.Label).join(cls).filter(*filters)

        for label in query:
            session.delete(label)
        if commit:
            session.commit()

    @staticmethod
    def _transform_schedule_model_to_scheme(
        db_schedule: Schedule,
    ) -> schemas.ScheduleRecord:
        schedule = schemas.ScheduleRecord.from_orm(db_schedule)
        SQLDB._add_utc_timezone(schedule, "creation_time")
        return schedule

    @staticmethod
    def _add_utc_timezone(obj, attribute_name):
        """
        sqlalchemy losing timezone information with sqlite so we're returning it
        https://stackoverflow.com/questions/6991457/sqlalchemy-losing-timezone-information-with-sqlite
        """
        setattr(obj, attribute_name, pytz.utc.localize(getattr(obj, attribute_name)))

    @staticmethod
    def _transform_feature_set_model_to_schema(
        feature_set_record: FeatureSet, tag=None,
    ) -> schemas.FeatureSet:
        feature_set_full_dict = feature_set_record.full_object
        feature_set_resp = schemas.FeatureSet(**feature_set_full_dict)

        feature_set_resp.metadata.tag = tag
        return feature_set_resp
