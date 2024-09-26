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
#
import asyncio
import collections
import functools
import hashlib
import pathlib
import re
import typing
import urllib.parse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import fastapi.concurrency
import mergedeep
import pytz
from sqlalchemy import MetaData, and_, case, delete, distinct, func, or_, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session, aliased

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.formatters
import mlrun.common.runtimes.constants
import mlrun.common.schemas
import mlrun.common.types
import mlrun.errors
import mlrun.k8s_utils
import mlrun.model
import server.api.db.session
import server.api.utils.helpers
from mlrun.artifacts.base import fill_artifact_object_hash
from mlrun.common.schemas.feature_store import (
    FeatureSetDigestOutputV2,
    FeatureSetDigestSpecV2,
)
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.lists import ArtifactList, RunList
from mlrun.model import RunObject
from mlrun.utils import (
    fill_function_hash,
    fill_object_hash,
    generate_artifact_uri,
    generate_object_uri,
    get_in,
    is_legacy_artifact,
    logger,
    update_in,
    validate_artifact_key_name,
    validate_tag_name,
)
from server.api.db.base import DBInterface
from server.api.db.sqldb.helpers import (
    MemoizationCache,
    generate_query_predicate_for_name,
    label_set,
    run_labels,
    run_start_time,
    run_state,
    update_labels,
)
from server.api.db.sqldb.models import (
    AlertConfig,
    AlertState,
    AlertTemplate,
    Artifact,
    ArtifactV2,
    BackgroundTask,
    Base,
    DatastoreProfile,
    DataVersion,
    Entity,
    Feature,
    FeatureSet,
    FeatureVector,
    Function,
    HubSource,
    Log,
    PaginationCache,
    Project,
    ProjectSummary,
    Run,
    Schedule,
    TimeWindowTracker,
    User,
    _labeled,
    _tagged,
    _with_notifications,
)

NULL = None  # Avoid flake8 issuing warnings when comparing in filter
unversioned_tagged_object_uid_prefix = "unversioned-"

conflict_messages = [
    "(sqlite3.IntegrityError) UNIQUE constraint failed",
    "(pymysql.err.IntegrityError) (1062",
    "(pymysql.err.IntegrityError) (1586",
]


def retry_on_conflict(function):
    """
    Most of our store_x functions starting from doing get, then if nothing is found creating the object otherwise
    updating attributes on the existing object. On the SQL level this translates to either INSERT or UPDATE queries.
    Sometimes we have a race condition in which two requests do the get, find nothing, create a new object, but only the
    SQL query of the first one will succeed, the second will get a conflict error, in that case, a retry like we're
    doing on the bottom most layer (like the one for the database is locked error) won't help, cause the object does not
    hold a reference to the existing DB object, and therefore will always translate to an INSERT query, therefore, in
    order to make it work, we need to do the get again, and in other words, call the whole store_x function again
    This why we implemented this retry as a decorator that comes "around" the existing functions
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        def _try_function():
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                if mlrun.utils.helpers.are_strings_in_exception_chain_messages(
                    exc, conflict_messages
                ):
                    logger.warning(
                        "Got conflict error from DB. Retrying", err=err_to_str(exc)
                    )
                    raise mlrun.errors.MLRunRuntimeError(
                        "Got conflict error from DB"
                    ) from exc
                raise mlrun.errors.MLRunFatalFailureError(original_exception=exc)

        if config.httpdb.db.conflict_retry_timeout:
            interval = config.httpdb.db.conflict_retry_interval
            if interval is None:
                interval = mlrun.utils.create_step_backoff([[0.0001, 1], [3, None]])
            return mlrun.utils.helpers.retry_until_successful(
                interval,
                config.httpdb.db.conflict_retry_timeout,
                logger,
                False,
                _try_function,
            )
        else:
            return function(*args, **kwargs)

    return wrapper


class SQLDB(DBInterface):
    def __init__(self, dsn=""):
        self.dsn = dsn
        self._name_with_iter_regex = re.compile("^[0-9]+-.+$")

    def initialize(self, session):
        if self.dsn and self.dsn.startswith("sqlite:///"):
            logger.info("Creating sqlite db file", dsn=self.dsn)
            parsed = urllib.parse.urlparse(self.dsn)
            pathlib.Path(parsed.path[1:]).parent.mkdir(parents=True, exist_ok=True)

    # ---- Logs ----
    def store_log(
        self,
        session,
        uid,
        project="",
        body=b"",
        append=False,
    ):
        raise NotImplementedError("DB should not be used for logs storage")

    def get_log(self, session, uid, project="", offset=0, size=0):
        raise NotImplementedError("DB should not be used for logs storage")

    def delete_log(self, session: Session, project: str, uid: str):
        project = project or config.default_project
        self._delete(session, Log, project=project, uid=uid)

    def _delete_logs(self, session: Session, project: str):
        logger.debug("Removing logs from db", project=project)
        for log in self._list_logs(session, project):
            self.delete_log(session, project, log.uid)

    def _list_logs(self, session: Session, project: str):
        return self._query(session, Log, project=project).all()

    # ---- Runs ----
    @retry_on_conflict
    def store_run(
        self,
        session,
        run_data,
        uid,
        project="",
        iter=0,
    ):
        logger.debug(
            "Storing run to db",
            project=project,
            uid=uid,
            iter=iter,
            run_name=run_data["metadata"]["name"],
        )
        # Do not lock run as it may cause deadlocks
        run = self._get_run(session, uid, project, iter)
        now = datetime.now(timezone.utc)
        if not run:
            run = Run(
                name=run_data["metadata"]["name"],
                uid=uid,
                project=project,
                iteration=iter,
                state=run_state(run_data),
                start_time=run_start_time(run_data) or now,
                requested_logs=False,
            )
        self._ensure_run_name_on_update(run, run_data)
        labels = run_labels(run_data)
        self._update_run_state(run, run_data)
        update_labels(run, labels)
        # Note that this code basically allowing anyone to override the run's start time after it was already set
        # This is done to enable the context initialization to set the start time to when the user's code actually
        # started running, and not when the run record was initially created (happening when triggering the job)
        # In the future we might want to limit who can actually do that
        start_time = run_start_time(run_data) or SQLDB._add_utc_timezone(run.start_time)
        run_data.setdefault("status", {})["start_time"] = start_time.isoformat()
        run.start_time = start_time
        self._update_run_updated_time(run, run_data, now=now)
        run.struct = run_data
        self._upsert(session, [run], ignore=True)

    def update_run(self, session, updates: dict, uid, project="", iter=0):
        project = project or config.default_project
        run = self._get_run(session, uid, project, iter, with_for_update=True)
        if not run:
            run_uri = RunObject.create_uri(project, uid, iter)
            raise mlrun.errors.MLRunNotFoundError(f"Run {run_uri} not found")
        struct = run.struct
        for key, val in updates.items():
            update_in(struct, key, val)
        self._ensure_run_name_on_update(run, struct)
        self._update_run_state(run, struct)
        start_time = run_start_time(struct)
        if start_time:
            run.start_time = start_time

        # Update the labels only if the run updates contains labels
        if run_labels(updates):
            update_labels(run, run_labels(struct))
        self._update_run_updated_time(run, struct)
        run.struct = struct
        self._upsert(session, [run])
        self._delete_empty_labels(session, Run.Label)
        return run.struct

    def list_distinct_runs_uids(
        self,
        session,
        project: str = None,
        requested_logs_modes: list[bool] = None,
        only_uids=True,
        last_update_time_from: datetime = None,
        states: list[str] = None,
        specific_uids: list[str] = None,
    ) -> typing.Union[list[str], RunList]:
        """
        List all runs uids in the DB
        :param session: DB session
        :param project: Project name, `*` or `None` lists across all projects
        :param requested_logs_modes: If not `None`, will return only runs with the given requested logs modes
        :param only_uids: If True, will return only the uids of the runs as list of strings
                          If False, will return the full run objects as RunList
        :param last_update_time_from: If not `None`, will return only runs updated after this time
        :param states: If not `None`, will return only runs with the given states
        :return: List of runs uids or RunList
        """
        if only_uids:
            # using distinct to avoid duplicates as there could be multiple runs with the same uid(different iterations)
            query = self._query(session, distinct(Run.uid))
        else:
            query = self._query(session, Run)

        if project and project != "*":
            query = query.filter(Run.project == project)

        if states:
            query = query.filter(Run.state.in_(states))

        if last_update_time_from is not None:
            query = query.filter(Run.updated >= last_update_time_from)

        if requested_logs_modes is not None:
            query = query.filter(Run.requested_logs.in_(requested_logs_modes))

        if specific_uids:
            query = query.filter(Run.uid.in_(specific_uids))

        if not only_uids:
            # group_by allows us to have a row per uid with the whole record rather than just the uid (as distinct does)
            # note we cannot promise that the same row will be returned each time per uid as the order is not guaranteed
            query = query.group_by(Run.uid)

            runs = RunList()
            for run in query:
                runs.append(run.struct)

            return runs

        # from each row we expect to get a tuple of (uid,) so we need to extract the uid from the tuple
        return [uid for (uid,) in query.all()]

    def update_runs_requested_logs(
        self, session, uids: list[str], requested_logs: bool = True
    ):
        # note that you should commit right after the synchronize_session=False
        # https://stackoverflow.com/questions/70350298/what-does-synchronize-session-false-do-exactly-in-update-functions-for-sqlalch
        self._query(session, Run).filter(Run.uid.in_(uids)).update(
            {
                Run.requested_logs: requested_logs,
                Run.updated: datetime.now(timezone.utc),
            },
            synchronize_session=False,
        )
        session.commit()

    def read_run(self, session, uid, project=None, iter=0):
        project = project or config.default_project
        run = self._get_run(session, uid, project, iter)
        if not run:
            raise mlrun.errors.MLRunNotFoundError(
                f"Run uid {uid} of project {project} not found"
            )
        return run.struct

    def list_runs(
        self,
        session,
        name: typing.Optional[str] = None,
        uid: typing.Optional[typing.Union[str, list[str]]] = None,
        project: str = "",
        labels: typing.Optional[typing.Union[str, list[str]]] = None,
        states: typing.Optional[list[mlrun.common.runtimes.constants.RunStates]] = None,
        sort: bool = True,
        last: int = 0,
        iter: bool = False,
        start_time_from: datetime = None,
        start_time_to: datetime = None,
        last_update_time_from: datetime = None,
        last_update_time_to: datetime = None,
        partition_by: mlrun.common.schemas.RunPartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
        max_partitions: int = 0,
        requested_logs: bool = None,
        return_as_run_structs: bool = True,
        with_notifications: bool = False,
        page: typing.Optional[int] = None,
        page_size: typing.Optional[int] = None,
    ) -> RunList:
        project = project or config.default_project
        query = self._find_runs(session, uid, project, labels)
        if name is not None:
            query = self._add_run_name_query(query, name)
        if states is not None:
            query = query.filter(Run.state.in_(states))
        if start_time_from is not None:
            query = query.filter(Run.start_time >= start_time_from)
        if start_time_to is not None:
            query = query.filter(Run.start_time <= start_time_to)
        if last_update_time_from is not None:
            query = query.filter(Run.updated >= last_update_time_from)
        if last_update_time_to is not None:
            query = query.filter(Run.updated <= last_update_time_to)
        if sort:
            query = query.order_by(Run.start_time.desc())
        if last:
            if not sort:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Limiting the number of returned records without sorting will provide non-deterministic results"
                )
            query = query.limit(last)
        if not iter:
            query = query.filter(Run.iteration == 0)
        if requested_logs is not None:
            query = query.filter(Run.requested_logs == requested_logs)
        # Purposefully not using outer join to avoid returning runs without notifications
        if with_notifications:
            query = query.join(Run.Notification)
        if partition_by:
            self._assert_partition_by_parameters(
                mlrun.common.schemas.RunPartitionByField,
                partition_by,
                partition_sort_by,
            )
            query = self._create_partitioned_query(
                session,
                query,
                Run,
                partition_by,
                rows_per_partition,
                partition_sort_by,
                partition_order,
                max_partitions,
            )

        query = self._paginate_query(query, page, page_size)

        if not return_as_run_structs:
            return query.all()

        runs = RunList()
        for run in query:
            run_struct = run.struct
            if with_notifications:
                run_struct.setdefault("spec", {}).setdefault("notifications", [])
                run_struct.setdefault("status", {}).setdefault("notifications", {})
                for notification in run.notifications:
                    (
                        notification_spec,
                        notification_status,
                    ) = self._transform_notification_record_to_spec_and_status(
                        notification
                    )
                    run_struct["spec"]["notifications"].append(notification_spec)
                    run_struct["status"]["notifications"][notification.name] = (
                        notification_status
                    )
            runs.append(run_struct)

        return runs

    def del_run(self, session, uid, project=None, iter=0):
        project = project or config.default_project
        # We currently delete *all* iterations
        self._delete(session, Run, uid=uid, project=project)

    def del_runs(
        self, session, name=None, project=None, labels=None, state=None, days_ago=0
    ):
        project = project or config.default_project
        query = self._find_runs(session, None, project, labels)
        if days_ago:
            since = datetime.now(timezone.utc) - timedelta(days=days_ago)
            query = query.filter(Run.start_time >= since)
        if name:
            query = self._add_run_name_query(query, name)
        if state:
            query = query.filter(Run.state == state)
        for run in query:  # Can not use query.delete with join
            session.delete(run)
        session.commit()

    def _add_run_name_query(self, query, name):
        exact_name = self._escape_characters_for_like_query(name)
        if name.startswith("~"):
            query = query.filter(Run.name.ilike(f"%{exact_name[1:]}%", escape="\\"))
        else:
            query = query.filter(Run.name == name)
        return query

    @staticmethod
    def _ensure_run_name_on_update(run_record: Run, run_dict: dict):
        body_name = run_dict["metadata"]["name"]
        if body_name != run_record.name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Changing name for an existing run is invalid"
            )

    @staticmethod
    def _update_run_updated_time(
        run_record: Run, run_dict: dict, now: typing.Optional[datetime] = None
    ):
        if now is None:
            now = datetime.now(timezone.utc)
        run_record.updated = now
        run_dict.setdefault("status", {})["last_update"] = now.isoformat()

    @staticmethod
    def _update_run_state(run_record: Run, run_dict: dict):
        state = run_state(run_dict)
        run_record.state = state
        run_dict.setdefault("status", {})["state"] = state

    # ---- Artifacts ----
    @retry_on_conflict
    def store_artifact(
        self,
        session,
        key,
        artifact,
        uid=None,
        iter=None,
        tag="",
        project="",
        producer_id="",
        best_iteration=False,
        always_overwrite=False,
    ) -> str:
        project = project or config.default_project
        tag = tag or "latest"

        # handle link artifacts separately
        if artifact.get("kind") == mlrun.common.schemas.ArtifactCategories.link.value:
            return self._mark_best_iteration_artifact(
                session,
                project,
                key,
                artifact,
                uid,
            )

        if tag:
            # fail early if tag is invalid
            validate_tag_name(tag, "artifact.metadata.tag")

        original_uid = uid

        if isinstance(artifact, dict):
            artifact_dict = artifact
        else:
            artifact_dict = artifact.to_dict()

        if not artifact_dict.get("metadata", {}).get("key"):
            artifact_dict.setdefault("metadata", {})["key"] = key
        if not artifact_dict.get("metadata", {}).get("project"):
            artifact_dict.setdefault("metadata", {})["project"] = project

        # calculate uid
        uid = fill_artifact_object_hash(artifact_dict, iter, producer_id)

        # If object was referenced by UID, the request cannot modify it
        if original_uid and uid != original_uid:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Changing uid for an object referenced by its uid"
            )

        # for easier querying, we mark artifacts without iteration as best iteration
        if not best_iteration and (iter is None or iter == 0):
            best_iteration = True

        # try to get an existing artifact with the same calculated uid
        existing_artifact = self._get_existing_artifact(
            session, project, key, uid, producer_id=producer_id, iteration=iter
        )

        # if the object is not new, we need to check if we need to update it or create a new one
        if existing_artifact:
            if (
                self._should_update_artifact(existing_artifact, uid, iter)
                or always_overwrite
            ):
                logger.debug(
                    "Updating an existing artifact",
                    project=project,
                    key=key,
                    iteration=iter,
                    uid=uid,
                )
                db_artifact = existing_artifact
                self._update_artifact_record_from_dict(
                    db_artifact,
                    artifact_dict,
                    project,
                    key,
                    uid,
                    iter,
                    best_iteration,
                    producer_id,
                )
                self._upsert(session, [db_artifact])
                if tag:
                    self.tag_artifacts(session, tag, [db_artifact], project)
                return uid
            logger.debug(
                "A similar artifact exists, but some values have changed - creating a new artifact",
                project=project,
                key=key,
                iteration=iter,
                producer_id=producer_id,
            )

        return self.create_artifact(
            session,
            project,
            artifact_dict,
            key,
            tag,
            uid,
            iter,
            producer_id,
            best_iteration,
        )

    def create_artifact(
        self,
        session,
        project,
        artifact,
        key,
        tag="",
        uid=None,
        iteration=None,
        producer_id="",
        best_iteration=False,
    ):
        if not uid:
            uid = fill_artifact_object_hash(artifact, iteration, producer_id)

        # check if the object already exists
        query = self._query(session, ArtifactV2, key=key, project=project, uid=uid)
        existing_object = query.one_or_none()
        if existing_object:
            object_uri = generate_object_uri(project, key, tag)
            raise mlrun.errors.MLRunConflictError(
                f"Adding an already-existing {ArtifactV2.__name__} - {object_uri}"
            )

        validate_artifact_key_name(key, "artifact.key")

        db_artifact = ArtifactV2(project=project, key=key)
        self._update_artifact_record_from_dict(
            db_artifact,
            artifact,
            project,
            key,
            uid,
            iteration,
            best_iteration,
            producer_id,
        )

        self._upsert(session, [db_artifact])
        if tag:
            validate_tag_name(tag, "artifact.metadata.tag")
            self.tag_artifacts(
                session,
                tag,
                [db_artifact],
                project,
            )

        # we want to tag the artifact also as "latest" if it's the first time we store it
        if tag != "latest":
            self.tag_artifacts(session, "latest", [db_artifact], project)

        return uid

    def list_artifacts(
        self,
        session,
        name=None,
        project=None,
        tag=None,
        labels=None,
        since: datetime = None,
        until: datetime = None,
        kind=None,
        category: mlrun.common.schemas.ArtifactCategories = None,
        iter: int = None,
        best_iteration: bool = False,
        as_records: bool = False,
        uid: str = None,
        producer_id: str = None,
        producer_uri: str = None,
        most_recent: bool = False,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
        limit: int = None,
    ):
        project = project or config.default_project

        if best_iteration and iter is not None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Best iteration cannot be used when iter is specified"
            )

        artifact_records = self._find_artifacts(
            session,
            project,
            tag=tag,
            labels=labels,
            since=since,
            until=until,
            name=name,
            kind=kind,
            category=category,
            iter=iter,
            uid=uid,
            producer_id=producer_id,
            best_iteration=best_iteration,
            most_recent=most_recent,
            attach_tags=not as_records,
            limit=limit,
        )
        if as_records:
            return artifact_records

        artifacts = ArtifactList()
        for artifact, artifact_tag in artifact_records:
            artifact_struct = artifact.full_object

            # TODO: filtering by producer uri may be a heavy operation when there are many artifacts in a workflow.
            #  We should filter the artifacts before loading them into memory with query.all()
            # Producer URI usually points to a run and is used to filter artifacts by the run that produced them.
            # When the artifact was produced by a workflow, the producer id is a workflow id.
            if producer_uri:
                artifact_struct.setdefault("spec", {}).setdefault("producer", {})
                artifact_producer_uri = artifact_struct["spec"]["producer"].get(
                    "uri", None
                )
                # We check if the producer uri is a substring of the artifact producer uri because it
                # may contain additional information (like the run iteration) that we don't want to filter by.
                if (
                    artifact_producer_uri is not None
                    and producer_uri not in artifact_producer_uri
                ):
                    continue

            self._set_tag_in_artifact_struct(artifact_struct, artifact_tag)
            artifacts.append(
                mlrun.common.formatters.ArtifactFormat.format_obj(
                    artifact_struct, format_
                )
            )

        return artifacts

    def list_artifacts_for_producer_id(
        self,
        session,
        producer_id: str,
        project: str = None,
        key_tag_iteration_pairs: list[tuple] = "",
    ) -> ArtifactList:
        project = project or mlrun.mlconf.default_project
        artifact_records = self._find_artifacts_for_producer_id(
            session,
            producer_id=producer_id,
            project=project,
            key_tag_iteration_pairs=key_tag_iteration_pairs,
        )

        artifacts = ArtifactList()
        for artifact, artifact_tag in artifact_records:
            artifact_struct = artifact.full_object
            self._set_tag_in_artifact_struct(artifact_struct, artifact_tag)
            artifacts.append(artifact_struct)

        return artifacts

    def read_artifact(
        self,
        session,
        key: str,
        tag: str = None,
        iter: int = None,
        project: str = None,
        producer_id: str = None,
        uid: str = None,
        raise_on_not_found: bool = True,
        format_: mlrun.common.formatters.ArtifactFormat = mlrun.common.formatters.ArtifactFormat.full,
    ):
        query = self._query(session, ArtifactV2, key=key, project=project)
        enrich_tag = False

        if uid:
            query = query.filter(ArtifactV2.uid == uid)
        if producer_id:
            query = query.filter(ArtifactV2.producer_id == producer_id)

        if tag == "latest" and uid:
            # Make a best-effort attempt to find the "latest" tag. It will be present in the response if the
            # latest tag exists, otherwise, it will not be included.
            # This is due to 'latest' being a special case and is enriched in the client side
            latest_query = query.join(
                ArtifactV2.Tag, ArtifactV2.Tag.obj_id == ArtifactV2.id
            ).filter(ArtifactV2.Tag.name == "latest")
            if latest_query.one_or_none():
                enrich_tag = True
        elif tag:
            # If a specific tag is provided, handle all cases where UID may or may not be included.
            # The case for UID with the "latest" tag is already covered above.
            # Here, we join with the tags table to check for a match with the specified tag.
            enrich_tag = True
            query = query.join(
                ArtifactV2.Tag, ArtifactV2.Tag.obj_id == ArtifactV2.id
            ).filter(ArtifactV2.Tag.name == tag)

        # keep the query without the iteration filter for later error handling
        query_without_iter = query
        if iter is not None:
            query = query.filter(ArtifactV2.iteration == iter)

        db_artifact = query.one_or_none()

        if not db_artifact:
            # if the artifact was not found and iter==0, we might be looking for a link artifact
            # in this case, we need to look for the artifact with the best iteration
            fail = True
            if iter == 0:
                query_without_iter = query_without_iter.filter(
                    ArtifactV2.best_iteration
                )
                db_artifact = query_without_iter.one_or_none()
                if db_artifact is not None:
                    # we found something, so we can continue
                    fail = False

            if fail:
                if raise_on_not_found:
                    artifact_uri = generate_artifact_uri(project, key, tag, iter)
                    raise mlrun.errors.MLRunNotFoundError(
                        f"Artifact {artifact_uri} not found"
                    )
                return None

        artifact = db_artifact.full_object

        # If connected to a tag add it to metadata
        if enrich_tag:
            self._set_tag_in_artifact_struct(artifact, tag)

        return mlrun.common.formatters.ArtifactFormat.format_obj(artifact, format_)

    def del_artifact(
        self, session, key, tag="", project="", uid=None, producer_id=None, iter=None
    ):
        project = project or config.default_project
        self._delete_tagged_object(
            session,
            ArtifactV2,
            project=project,
            tag=tag,
            uid=uid,
            key=key,
            producer_id=producer_id,
            iteration=iter,
        )

    def del_artifacts(
        self,
        session,
        name="",
        project="",
        tag="*",
        labels=None,
        ids=None,
        producer_id=None,
    ):
        project = project or config.default_project
        distinct_keys_and_uids = self._find_artifacts(
            session=session,
            project=project,
            name=name,
            ids=ids,
            tag=tag,
            labels=labels,
            producer_id=producer_id,
            with_entities=[ArtifactV2.key, ArtifactV2.uid],
        )

        artifact_column_identifiers = {}
        for key, uid in distinct_keys_and_uids:
            artifact_column_identifier, column_value = self._delete_tagged_object(
                session,
                ArtifactV2,
                project=project,
                uid=uid,
                key=key,
                commit=False,
                producer_id=producer_id,
            )
            if artifact_column_identifier is None:
                # record was not found
                continue

            artifact_column_identifiers.setdefault(
                artifact_column_identifier, []
            ).append(column_value)

        failed_deletions_count = 0
        for (
            artifact_column_identifier,
            column_values,
        ) in artifact_column_identifiers.items():
            deletions_count = self._delete_multi_objects(
                session=session,
                main_table=ArtifactV2,
                related_tables=[ArtifactV2.Tag, ArtifactV2.Label],
                project=project,
                main_table_identifier=getattr(ArtifactV2, artifact_column_identifier),
                main_table_identifier_values=column_values,
            )
            failed_deletions_count += len(column_values) - deletions_count

        if failed_deletions_count:
            raise mlrun.errors.MLRunInternalServerError(
                f"Failed to delete {failed_deletions_count} artifacts"
            )

    def list_artifact_tags(
        self, session, project, category: mlrun.common.schemas.ArtifactCategories = None
    ) -> list[str]:
        """
        List all tags for artifacts in the DB

        :param session: DB session
        :param project: Project name
        :param category: Artifact category to filter by

        :return: a list of distinct tags
        """
        query = (
            self._query(session, ArtifactV2.Tag.name)
            .select_from(ArtifactV2)
            .join(ArtifactV2.Tag, ArtifactV2.Tag.obj_id == ArtifactV2.id)
            .filter(ArtifactV2.project == project)
            .group_by(ArtifactV2.Tag.name)
        )
        if category:
            query = self._add_artifact_category_query(category, query).with_hint(
                ArtifactV2, "USE INDEX (idx_project_kind)"
            )

        # the query returns a list of tuples, we need to extract the tag from each tuple
        return [tag for (tag,) in query]

    @retry_on_conflict
    def overwrite_artifacts_with_tag(
        self,
        session: Session,
        project: str,
        tag: str,
        identifiers: list[mlrun.common.schemas.ArtifactIdentifier],
    ):
        # query all artifacts which match the identifiers
        artifacts = []
        for identifier in identifiers:
            artifacts += self._list_artifacts_for_tagging(
                session,
                project_name=project,
                identifier=identifier,
            )

        # TODO: remove duplicates artifacts entries

        # delete related tags from artifacts identifiers
        # not committing the session here because we want to do it atomic with the next query
        self._delete_artifacts_tags(session, project, artifacts, commit=False)

        # tag artifacts with tag
        self.tag_artifacts(session, tag, artifacts, project)

    @retry_on_conflict
    def append_tag_to_artifacts(
        self,
        session: Session,
        project: str,
        tag: str,
        identifiers: list[mlrun.common.schemas.ArtifactIdentifier],
    ):
        # query all artifacts which match the identifiers
        artifacts = []
        for identifier in identifiers:
            artifacts += self._list_artifacts_for_tagging(
                session,
                project_name=project,
                identifier=identifier,
            )
        self.tag_artifacts(session, tag, artifacts, project)

    def delete_tag_from_artifacts(
        self,
        session: Session,
        project: str,
        tag: str,
        identifiers: list[mlrun.common.schemas.ArtifactIdentifier],
    ):
        # query all artifacts which match the identifiers
        artifacts = []
        for identifier in identifiers:
            artifacts += self._list_artifacts_for_tagging(
                session,
                project_name=project,
                identifier=identifier,
            )
        self._delete_artifacts_tags(session, project, artifacts, tags=[tag])

    def tag_artifacts(
        self,
        session,
        tag_name: str,
        artifacts,
        project: str,
    ):
        artifacts_keys = [artifact.key for artifact in artifacts]
        if not artifacts_keys:
            logger.debug(
                "No artifacts to tag",
                project=project,
                tag=tag_name,
                artifacts=artifacts,
            )
            return

        logger.debug(
            "Locking artifacts in db before tagging artifacts",
            project=project,
            tag=tag_name,
            artifacts_keys=artifacts_keys,
        )

        # to avoid multiple runs trying to tag the same artifacts simultaneously,
        # lock the artifacts with the same keys for the entire transaction (using with_for_update).
        self._query(
            session,
            ArtifactV2,
            project=project,
        ).with_entities(ArtifactV2.id).filter(
            ArtifactV2.key.in_(artifacts_keys),
        ).order_by(ArtifactV2.id.asc()).populate_existing().with_for_update().all()

        logger.debug(
            "Acquired artifacts db lock",
            project=project,
            tag=tag_name,
            artifacts_keys=artifacts_keys,
        )

        objects = []
        for artifact in artifacts:
            # remove the tags of the same name that point to artifacts with the same key
            # and a different producer id
            query = (
                self._query(
                    session,
                    artifact.Tag,
                    name=tag_name,
                    project=project,
                    obj_name=artifact.key,
                )
                .join(
                    ArtifactV2,
                )
                .filter(
                    ArtifactV2.producer_id != artifact.producer_id,
                )
            )

            # delete the tags
            for old_tag in query:
                objects.append(old_tag)
                session.delete(old_tag)

            def _get_tag(_session):
                # search for an existing tag with the same name, and points to artifacts with the same key, producer id,
                # and iteration. this means that the same producer created this artifact,
                # and we can update the existing tag
                tag_query = (
                    self._query(
                        _session,
                        artifact.Tag,
                        name=tag_name,
                        project=project,
                        obj_name=artifact.key,
                    )
                    .join(
                        ArtifactV2,
                    )
                    .filter(
                        ArtifactV2.producer_id == artifact.producer_id,
                        ArtifactV2.iteration == artifact.iteration,
                    )
                )

                return tag_query.one_or_none()

            # to make sure we can list tags that were created during this session in parallel by different processes,
            # we need to use a new session. if there is an existing tag, we'll definitely get it, so we can update it
            # instead of creating a new tag.
            tag = server.api.db.session.run_function_with_new_db_session(_get_tag)
            if not tag:
                # create the new tag
                tag = artifact.Tag(
                    project=project,
                    name=tag_name,
                    obj_name=artifact.key,
                )
            tag.obj_id = artifact.id

            objects.append(tag)
            session.add(tag)

        # commit the changes, including the deletion of the old tags and the creation of the new tags
        # this will also release the locks on the artifacts' rows
        self._commit(session, objects)

        logger.debug(
            "Released artifacts db lock after tagging artifacts",
            project=project,
            tag=tag_name,
            artifacts_keys=artifacts_keys,
        )

    def _mark_best_iteration_artifact(
        self,
        session,
        project,
        key,
        link_artifact,
        uid=None,
    ):
        artifacts_to_commit = []

        # get the artifact record from the db
        link_iteration = link_artifact.get("spec", {}).get("link_iteration")
        link_tree = link_artifact.get("spec", {}).get("link_tree") or link_artifact.get(
            "metadata", {}
        ).get("tree")
        link_key = link_artifact.get("spec", {}).get("link_key")
        if link_key:
            key = link_key

        # Lock the artifacts with the same project and key (and producer_id when available) to avoid unexpected
        # deadlocks and conform to our lock-once-when-starting logic - ML-6869
        lock_query = self._query(
            session,
            ArtifactV2,
            project=project,
            key=key,
        ).with_entities(ArtifactV2.id)
        if link_tree:
            lock_query = lock_query.filter(ArtifactV2.producer_id == link_tree)

        lock_query.order_by(
            ArtifactV2.id.asc()
        ).populate_existing().with_for_update().all()

        # get the best iteration artifact record
        query = self._query(session, ArtifactV2).filter(
            ArtifactV2.project == project,
            ArtifactV2.key == key,
            ArtifactV2.iteration == link_iteration,
        )
        if link_tree:
            query = query.filter(ArtifactV2.producer_id == link_tree)
        if uid:
            query = query.filter(ArtifactV2.uid == uid)

        best_iteration_artifact_record = query.one_or_none()
        if not best_iteration_artifact_record:
            raise mlrun.errors.MLRunNotFoundError(
                f"Best iteration artifact not found - {project}/{key}:{link_iteration}",
            )

        # get the previous best iteration artifact
        query = self._query(session, ArtifactV2).filter(
            ArtifactV2.project == project,
            ArtifactV2.key == key,
            ArtifactV2.best_iteration,
            ArtifactV2.iteration != link_iteration,
        )
        if link_tree:
            query = query.filter(ArtifactV2.producer_id == link_tree)

        previous_best_iteration_artifacts = query.one_or_none()
        if previous_best_iteration_artifacts:
            # remove the previous best iteration flag
            previous_best_iteration_artifacts.best_iteration = False
            artifacts_to_commit.append(previous_best_iteration_artifacts)

        # update the artifact record with best iteration
        best_iteration_artifact_record.best_iteration = True
        artifacts_to_commit.append(best_iteration_artifact_record)

        self._upsert(session, artifacts_to_commit)

        return best_iteration_artifact_record.uid

    def _update_artifact_record_from_dict(
        self,
        artifact_record,
        artifact_dict: dict,
        project: str,
        key: str,
        uid: str,
        iter: int = None,
        best_iteration: bool = False,
        producer_id: str = None,
    ):
        artifact_record.project = project
        kind = artifact_dict.get("kind") or "artifact"
        artifact_record.kind = kind
        artifact_record.producer_id = producer_id or artifact_dict["metadata"].get(
            "tree"
        )
        updated_datetime = datetime.now(timezone.utc)
        artifact_record.updated = updated_datetime
        created = (
            str(artifact_record.created)
            if artifact_record.created
            else artifact_dict["metadata"].pop("created", None)
        )
        # make sure we have a datetime object with timezone both in the artifact record and in the artifact dict
        created_datetime = mlrun.utils.enrich_datetime_with_tz_info(
            created
        ) or datetime.now(timezone.utc)
        artifact_record.created = created_datetime

        # if iteration is not given, we assume it is a single iteration artifact, and thus we set the iteration to 0
        artifact_record.iteration = iter or 0
        if best_iteration or iter == 0:
            artifact_record.best_iteration = True

        artifact_record.uid = uid

        artifact_dict["metadata"]["updated"] = str(updated_datetime)
        artifact_dict["metadata"]["created"] = str(created_datetime)
        artifact_dict["kind"] = kind

        db_key = artifact_dict.get("spec", {}).get("db_key")
        if not db_key:
            artifact_dict.setdefault("spec", {})["db_key"] = key

        # remove the tag from the metadata, as it is stored in a separate table
        artifact_dict["metadata"].pop("tag", None)

        artifact_record.full_object = artifact_dict

        # labels are stored in a separate table
        labels = artifact_dict["metadata"].pop("labels", {}) or {}
        update_labels(artifact_record, labels)

    def _list_artifacts_for_tagging(
        self,
        session: Session,
        project_name: str,
        identifier: mlrun.common.schemas.ArtifactIdentifier,
    ):
        artifacts = self.list_artifacts(
            session,
            project=project_name,
            name=identifier.key,
            kind=identifier.kind,
            iter=identifier.iter,
            uid=identifier.uid,
            producer_id=identifier.producer_id,
            as_records=True,
        )

        # in earlier versions, the uid actually stored the producer id of the artifacts, so in case we didn't find
        # any artifacts we should try to look for artifacts with the given uid as producer id
        if not artifacts and identifier.uid and not identifier.producer_id:
            artifacts = self.list_artifacts(
                session,
                project=project_name,
                name=identifier.key,
                kind=identifier.kind,
                iter=identifier.iter,
                producer_id=identifier.uid,
                as_records=True,
            )

        return artifacts

    @staticmethod
    def _set_tag_in_artifact_struct(artifact, tag):
        artifact["metadata"]["tag"] = tag

    def _get_link_artifacts_by_keys_and_uids(self, session, project, identifiers):
        # identifiers are tuples of (key, uid)
        if not identifiers:
            return []
        predicates = [
            and_(Artifact.key == key, Artifact.uid == uid) for (key, uid) in identifiers
        ]
        return (
            self._query(session, Artifact, project=project)
            .filter(or_(*predicates))
            .all()
        )

    def _delete_artifacts_tags(
        self,
        session,
        project: str,
        artifacts: list[ArtifactV2],
        tags: list[str] = None,
        commit: bool = True,
    ):
        artifacts_ids = [artifact.id for artifact in artifacts]
        query = session.query(ArtifactV2.Tag).filter(
            ArtifactV2.Tag.project == project,
            ArtifactV2.Tag.obj_id.in_(artifacts_ids),
        )
        if tags:
            query = query.filter(ArtifactV2.Tag.name.in_(tags))
        for tag in query:
            session.delete(tag)
        if commit:
            session.commit()

    def _find_artifacts(
        self,
        session: Session,
        project: str,
        ids: typing.Union[list[str], str] = None,
        tag: str = None,
        labels: typing.Union[list[str], str] = None,
        since: datetime = None,
        until: datetime = None,
        name: str = None,
        kind: mlrun.common.schemas.ArtifactCategories = None,
        category: mlrun.common.schemas.ArtifactCategories = None,
        iter: int = None,
        uid: str = None,
        producer_id: str = None,
        best_iteration: bool = False,
        most_recent: bool = False,
        attach_tags: bool = False,
        limit: int = None,
        with_entities: list[Any] = None,
    ) -> typing.Union[list[Any],]:
        """
        Find artifacts by the given filters.

        :param session: DB session
        :param project: Project name
        :param ids: Artifact IDs to filter by
        :param tag: Tag to filter by
        :param labels: Labels to filter by
        :param since: Filter artifacts that were updated after this time
        :param until: Filter artifacts that were updated before this time
        :param name: Artifact name to filter by
        :param kind: Artifact kind to filter by
        :param category: Artifact category to filter by (if kind is not given)
        :param iter: Artifact iteration to filter by
        :param uid: Artifact UID to filter by
        :param producer_id: Artifact producer ID to filter by
        :param best_iteration: Filter by best iteration artifacts
        :param most_recent: Filter by most recent artifacts
        :param attach_tags: Whether to return a list of tuples of (ArtifactV2, tag_name). If False, only ArtifactV2
        :param limit: Maximum number of artifacts to return
        :param with_entities: List of columns to return

        :return: May return:
            1. a list of tuples of (ArtifactV2, tag_name)
            2. a list of ArtifactV2 - if attach_tags is False
            3. a list of unique columns sets - if with_entities is given
        """
        if category and kind:
            message = "Category and Kind filters can't be given together"
            logger.warning(message, kind=kind, category=category)
            raise ValueError(message)

        # create a sub query that gets only the artifact IDs
        # apply all filters and limits
        query = session.query(ArtifactV2).with_entities(
            ArtifactV2.id,
            ArtifactV2.Tag.name,
        )

        if project:
            query = query.filter(ArtifactV2.project == project)
        if ids and ids != "*":
            query = query.filter(ArtifactV2.id.in_(ids))
        if uid:
            query = query.filter(ArtifactV2.uid == uid)
        if name:
            query = self._add_artifact_name_query(query, name)
        if iter is not None:
            query = query.filter(ArtifactV2.iteration == iter)
        if best_iteration:
            query = query.filter(ArtifactV2.best_iteration == best_iteration)
        if producer_id:
            query = query.filter(ArtifactV2.producer_id == producer_id)
        if labels:
            labels = label_set(labels)
            query = self._add_labels_filter(session, query, ArtifactV2, labels)
        if since or until:
            since = since or datetime.min
            until = until or datetime.max
            query = query.filter(
                and_(ArtifactV2.updated >= since, ArtifactV2.updated <= until)
            )
        if kind:
            query = query.filter(ArtifactV2.kind == kind)
        elif category:
            query = self._add_artifact_category_query(category, query)
        if most_recent:
            query = self._attach_most_recent_artifact_query(session, query)

        # join on tags
        if tag and tag != "*":
            # If a tag is given, we can just join (faster than outer join) and filter on the tag
            query = query.join(ArtifactV2.Tag, ArtifactV2.Tag.obj_id == ArtifactV2.id)
            query = query.filter(ArtifactV2.Tag.name == tag)
        else:
            # If no tag is given, we need to outer join to get all artifacts, even if they don't have tags
            query = query.outerjoin(
                ArtifactV2.Tag, ArtifactV2.Tag.obj_id == ArtifactV2.id
            )

        if limit:
            query = query.limit(limit)

        # limit operation loads all the results before performing the actual limiting,
        # therefore, we compile the above query as a sub query only for filtering out the relevant ids,
        # then join the outer query on the subquery to select the correct columns of the table.
        subquery = query.subquery()
        outer_query = session.query(ArtifactV2, subquery.c.name)
        if with_entities:
            outer_query = outer_query.with_entities(*with_entities, subquery.c.name)

        outer_query = outer_query.join(subquery, ArtifactV2.id == subquery.c.id)

        results = outer_query.all()
        if not attach_tags:
            # we might have duplicate records due to the tagging mechanism, so we need to deduplicate
            artifacts = set()
            for *artifact, _ in results:
                artifacts.add(tuple(artifact) if with_entities else artifact[0])

            return list(artifacts)

        return results

    def _find_artifacts_for_producer_id(
        self,
        session: Session,
        producer_id: str,
        project: str,
        key_tag_iteration_pairs: list[tuple] = "",
    ) -> list[tuple[ArtifactV2, str]]:
        """
        Find a producer's artifacts matching the given (key, tag, iteration) tuples.
        :param session:                 DB session
        :param producer_id:             The artifact producer ID to filter by
        :param project:                 Project name to filter by
        :param key_tag_iteration_pairs: List of tuples of (key, tag, iteration)
        :return: A list of tuples of (ArtifactV2, tag_name)
        """
        query = session.query(ArtifactV2, ArtifactV2.Tag.name)
        if project:
            query = query.filter(ArtifactV2.project == project)
        if producer_id:
            query = query.filter(ArtifactV2.producer_id == producer_id)

        query = query.join(ArtifactV2.Tag, ArtifactV2.Tag.obj_id == ArtifactV2.id)

        tuples_filter = []
        for key, tag, iteration in key_tag_iteration_pairs:
            iteration = iteration or 0
            tag = tag or "latest"
            tuples_filter.append(
                (ArtifactV2.key == key)
                & (ArtifactV2.Tag.name == tag)
                & (ArtifactV2.iteration == iteration)
            )

        query = query.filter(or_(*tuples_filter))
        return query.all()

    def _add_artifact_name_query(self, query, name=None):
        if not name:
            return query

        if name.startswith("~"):
            # Escape special chars (_,%) since we still need to do a like query.
            exact_name = self._escape_characters_for_like_query(name)
            # Use Like query to find substring matches
            return query.filter(
                ArtifactV2.key.ilike(f"%{exact_name[1:]}%", escape="\\")
            )

        return query.filter(ArtifactV2.key == name)

    @staticmethod
    def _add_artifact_category_query(category, query):
        kinds, exclude = category.to_kinds_filter()
        if exclude:
            query = query.filter(ArtifactV2.kind.notin_(kinds))
        else:
            query = query.filter(ArtifactV2.kind.in_(kinds))
        return query

    def _get_existing_artifact(
        self,
        session,
        project: str,
        key: str,
        uid: str = None,
        producer_id: str = None,
        iteration: int = None,
    ):
        query = self._query(session, ArtifactV2, key=key, project=project)
        if uid:
            query = query.filter(ArtifactV2.uid == uid)
        if producer_id:
            query = query.filter(ArtifactV2.producer_id == producer_id)
        if iteration is not None:
            query = query.filter(ArtifactV2.iteration == iteration)
        return query.one_or_none()

    def _should_update_artifact(
        self,
        existing_artifact: ArtifactV2,
        uid=None,
        iteration=None,
    ):
        # we should create a new artifact if we got a new iteration or the calculated uid is different.
        # otherwise we should update the existing artifact
        if uid is not None and existing_artifact.uid != uid:
            return False
        if iteration is not None and existing_artifact.iteration != iteration:
            return False
        return True

    def store_artifact_v1(
        self,
        session,
        key,
        artifact,
        uid,
        iter=None,
        tag="",
        project="",
        tag_artifact=True,
    ):
        """
        Store artifact v1 in the DB, this is the deprecated legacy artifact format
        and is only left for testing purposes
        """

        def _get_artifact(uid_, project_, key_):
            try:
                resp = self._query(
                    session, Artifact, uid=uid_, project=project_, key=key_
                ).one_or_none()
                return resp
            finally:
                pass

        project = project or config.default_project
        artifact = deepcopy(artifact)
        if is_legacy_artifact(artifact):
            updated, key, labels = self._process_legacy_artifact_v1_dict_to_store(
                artifact, key, iter
            )
        else:
            updated, key, labels = self._process_artifact_v1_dict_to_store(
                artifact, key, iter
            )
        existed = True
        art = _get_artifact(uid, project, key)
        if not art:
            # for backwards compatibility only validating key name on new artifacts
            validate_artifact_key_name(key, "artifact.key")
            art = Artifact(key=key, uid=uid, updated=updated, project=project)
            existed = False

        update_labels(art, labels)

        art.struct = artifact
        self._upsert(session, [art])
        if tag_artifact:
            tag = tag or "latest"

            # we want to ensure that the tag is valid before storing,
            # if it isn't, MLRunInvalidArgumentError will be raised
            validate_tag_name(tag, "artifact.metadata.tag")
            self._tag_artifacts_v1(session, [art], project, tag)
            # we want to tag the artifact also as "latest" if it's the first time we store it, reason is that there are
            # updates we are doing to the metadata of the artifact (like updating the labels) and we don't want those
            # changes to be reflected in the "latest" tag, as this in not actual the "latest" version of the artifact
            # which was produced by the user
            if not existed and tag != "latest":
                self._tag_artifacts_v1(session, [art], project, "latest")

    def read_artifact_v1(self, session, key, tag="", iter=None, project=""):
        """
        Read artifact v1 from the DB, this is the deprecated legacy artifact format
        """

        def _resolve_tag(cls, project_, name):
            ids = []
            for tag in self._query(session, cls.Tag, project=project_, name=name):
                ids.append(tag.obj_id)
            if not ids:
                return name  # Not found, return original uid
            return ids

        project = project or config.default_project
        ids = _resolve_tag(Artifact, project, tag)
        if iter:
            key = f"{iter}-{key}"

        query = self._query(session, Artifact, key=key, project=project)

        # This will hold the real tag of the object (if exists). Will be placed in the artifact structure.
        db_tag = None

        # TODO: refactor this
        # tag has 2 meanings:
        # 1. tag - in this case _resolve_tag will find the relevant uids and will return a list
        # 2. uid - in this case _resolve_tag won't find anything and simply return what was given to it, which actually
        # represents the uid
        if isinstance(ids, list) and ids:
            query = query.filter(Artifact.id.in_(ids))
            db_tag = tag
        elif isinstance(ids, str) and ids:
            query = query.filter(Artifact.uid == ids)
        else:
            # Select by last updated
            max_updated = session.query(func.max(Artifact.updated)).filter(
                Artifact.project == project, Artifact.key == key
            )
            query = query.filter(Artifact.updated.in_(max_updated))

        art = query.one_or_none()
        if not art:
            artifact_uri = generate_artifact_uri(project, key, tag, iter)
            raise mlrun.errors.MLRunNotFoundError(f"Artifact {artifact_uri} not found")

        artifact_struct = art.struct
        # We only set a tag in the object if the user asked specifically for this tag.
        if db_tag:
            self._set_tag_in_artifact_struct(artifact_struct, db_tag)
        return artifact_struct

    def _tag_artifacts_v1(self, session, artifacts, project: str, name: str):
        # found a bug in here, which is being exposed for when have multi-param execution.
        # each artifact key is being concatenated with the key and the iteration, this is problematic in this query
        # because we are filtering by the key+iteration and not just the key ( which would require some regex )
        # it would be fixed as part of the refactoring of the new artifact table structure where we would have
        # column for iteration as well.
        for artifact in artifacts:
            query = (
                self._query(
                    session,
                    artifact.Tag,
                    project=project,
                    name=name,
                )
                .join(Artifact)
                .filter(Artifact.key == artifact.key)
            )
            tag = query.one_or_none()
            if not tag:
                # To maintain backwards compatibility,
                # we validate the tag name only if it does not already exist on the artifact,
                # we don't want to fail on old tags that were created before the validation was added.
                validate_tag_name(tag_name=name, field_name="artifact.metadata.tag")
                tag = artifact.Tag(project=project, name=name)
            tag.obj_id = artifact.id
            self._upsert(session, [tag], ignore=True)

    @staticmethod
    def _process_artifact_v1_dict_to_store(artifact, key, iter=None):
        """
        This function is the deprecated is only left for testing purposes
        """
        updated = artifact["metadata"].get("updated")
        if not updated:
            updated = artifact["metadata"]["updated"] = datetime.now(timezone.utc)
        db_key = artifact["spec"].get("db_key")
        if db_key and db_key != key:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Conflict between requested key and key in artifact body"
            )
        if not db_key:
            artifact["spec"]["db_key"] = key
        if iter:
            key = f"{iter}-{key}"
        labels = artifact["metadata"].get("labels", {})

        # Ensure there is no "tag" field in the object, to avoid inconsistent situations between
        # body and tag parameter provided.
        artifact["metadata"].pop("tag", None)
        return updated, key, labels

    @staticmethod
    def _process_legacy_artifact_v1_dict_to_store(artifact, key, iter=None):
        """
        This function is the deprecated is only left for testing purposes
        """
        updated = artifact.get("updated")
        if not updated:
            updated = artifact["updated"] = datetime.now(timezone.utc)
        db_key = artifact.get("db_key")
        if db_key and db_key != key:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Conflict between requested key and key in artifact body"
            )
        if not db_key:
            artifact["db_key"] = key
        if iter:
            key = f"{iter}-{key}"
        labels = artifact.get("labels", {})

        # Ensure there is no "tag" field in the object, to avoid inconsistent situations between
        # body and tag parameter provided.
        artifact.pop("tag", None)
        return updated, key, labels

    # ---- Functions ----
    @retry_on_conflict
    def store_function(
        self,
        session,
        function,
        name,
        project="",
        tag="",
        versioned=False,
    ) -> str:
        logger.debug(
            "Storing function to DB",
            name=name,
            project=project,
            tag=tag,
            versioned=versioned,
            metadata=function.get("metadata"),
        )
        function = deepcopy(function)
        project = project or config.default_project
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
        body_name = function.get("metadata", {}).get("name")
        if body_name and body_name != name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Conflict between requested name and name in function body, function name is {name} while body_name is"
                f" {body_name}"
            )
        if not body_name:
            function.setdefault("metadata", {})["name"] = name
        if function_node_selector := get_in(function, "spec.node_selector"):
            mlrun.k8s_utils.validate_node_selectors(function_node_selector)
        fn = self._get_class_instance_by_uid(session, Function, name, project, uid)
        if not fn:
            fn = Function(
                name=name,
                project=project,
                uid=uid,
            )
        fn.updated = updated
        labels = get_in(function, "metadata.labels", {})
        update_labels(fn, labels)
        fn.struct = function
        self._upsert(session, [fn])
        self.tag_objects_v2(session, [fn], project, tag)
        return hash_key

    def list_functions(
        self,
        session: Session,
        name: typing.Optional[str] = None,
        project: typing.Optional[str] = None,
        tag: typing.Optional[str] = None,
        labels: list[str] = None,
        hash_key: typing.Optional[str] = None,
        format_: str = mlrun.common.formatters.FunctionFormat.full,
        page: typing.Optional[int] = None,
        page_size: typing.Optional[int] = None,
        since: datetime = None,
        until: datetime = None,
    ) -> list[dict]:
        project = project or mlrun.mlconf.default_project
        functions = []
        for function, function_tag in self._find_functions(
            session=session,
            name=name,
            project=project,
            labels=labels,
            tag=tag,
            hash_key=hash_key,
            since=since,
            until=until,
            page=page,
            page_size=page_size,
        ):
            function_dict = function.struct
            if not function_tag:
                # function status should be added only to tagged functions
                # TODO: remove explicit cleaning; we also
                #  will need to understand how to display functions in UI, because if we do not remove the status here,
                #  UI shows two function as `ready` which belong to the same Nuclio function
                function_dict["status"] = None

                # the unversioned uid is only a placeholder for tagged instances that are versioned.
                # if another instance "took" the tag, we're left with an unversioned untagged instance
                # don't list it
                if function.uid.startswith(unversioned_tagged_object_uid_prefix):
                    continue
            else:
                function_dict["metadata"]["tag"] = function_tag

            functions.append(
                mlrun.common.formatters.FunctionFormat.format_obj(
                    function_dict, format_
                )
            )
        return functions

    def get_function(
        self,
        session,
        name: str = None,
        project: str = None,
        tag: str = None,
        hash_key: str = None,
        format_: str = None,
    ) -> dict:
        """
        In version 1.4.0 we added a normalization to the function name before storing.
        To be backwards compatible and allow users to query old non-normalized functions,
        we're providing a fallback to get_function:
        normalize the requested name and try to retrieve it from the database.
        If no answer is received, we will check to see if the original name contained underscores,
        if so, the retrieval will be repeated and the result (if it exists) returned.
        """
        normalized_function_name = mlrun.utils.normalize_name(name)
        try:
            return self._get_function(
                session, normalized_function_name, project, tag, hash_key, format_
            )
        except mlrun.errors.MLRunNotFoundError as exc:
            if "_" in name:
                logger.warning(
                    "Failed to get underscore-named function, trying without normalization",
                    function_name=name,
                )
                return self._get_function(
                    session, name, project, tag, hash_key, format_
                )
            else:
                raise exc

    def delete_function(self, session: Session, project: str, name: str):
        logger.debug("Removing function from db", project=project, name=name)

        # deleting tags and labels, because in sqlite the relationships aren't necessarily cascading
        self._delete_function_tags(session, project, name, commit=False)
        self._delete_class_labels(
            session, Function, project=project, name=name, commit=False
        )
        self._delete(session, Function, project=project, name=name)

    def delete_functions(
        self, session: Session, project: str, names: typing.Union[str, list[str]]
    ) -> None:
        logger.debug("Removing functions from db", project=project, name=names)

        self._delete_multi_objects(
            session=session,
            main_table=Function,
            related_tables=[Function.Tag, Function.Label],
            project=project,
            main_table_identifier=Function.name,
            main_table_identifier_values=names,
        )

    def update_function(
        self,
        session,
        name,
        updates: dict,
        project: str = None,
        tag: str = "",
        hash_key: str = "",
    ):
        project = project or config.default_project
        query = self._query(session, Function, name=name, project=project)
        uid = self._get_function_uid(
            session=session, name=name, tag=tag, hash_key=hash_key, project=project
        )
        if uid:
            query = query.filter(Function.uid == uid)
        function = query.one_or_none()
        if function:
            struct = function.struct
            for key, val in updates.items():
                update_in(struct, key, val)
            function.struct = struct
            self._upsert(session, [function])
            return function.struct

    def update_function_external_invocation_url(
        self,
        session,
        name: str,
        url: str,
        project: str = "",
        tag: str = "",
        hash_key: str = "",
        operation: mlrun.common.types.Operation = mlrun.common.types.Operation.ADD,
    ):
        """
        This function updates the external invocation URLs of a function within a project.
        It can add or remove URLs based on the specified `operation` which can be
        either ADD or REMOVE of type :py:class:`~mlrun.types.Operation`
        """
        project = project or config.default_project
        normalized_function_name = mlrun.utils.normalize_name(name)
        function, _ = self._get_function_db_object(
            session,
            normalized_function_name,
            project,
            tag=tag or "latest",
            hash_key=hash_key,
        )
        if not function:
            logger.debug(
                "Function is not found, skipping external invocation urls update",
                project=project,
                name=name,
                url=url,
            )
            return

        # remove trailing slashes from the URL
        url = url.rstrip("/")

        struct = function.struct
        existing_invocation_urls = struct["status"].get("external_invocation_urls", [])
        updated = False
        if (
            operation == mlrun.common.types.Operation.ADD
            and url not in existing_invocation_urls
        ):
            logger.debug(
                "Adding new external invocation url to function",
                project=project,
                name=name,
                url=url,
            )
            updated = True
            existing_invocation_urls.append(url)
            struct["status"]["external_invocation_urls"] = existing_invocation_urls
        elif (
            operation == mlrun.common.types.Operation.REMOVE
            and url in existing_invocation_urls
        ):
            logger.debug(
                "Removing an external invocation url from function",
                project=project,
                name=name,
                url=url,
            )
            updated = True
            struct["status"]["external_invocation_urls"].remove(url)

        # update the function record only if the external invocation URLs were updated
        if updated:
            function.struct = struct
            self._upsert(session, [function])

    def _get_function(
        self,
        session,
        name: str = None,
        project: str = None,
        tag: str = None,
        hash_key: str = None,
        format_: str = mlrun.common.formatters.FunctionFormat.full,
    ):
        project = project or config.default_project
        computed_tag = tag or "latest"

        obj, uid = self._get_function_db_object(session, name, project, tag, hash_key)
        tag_function_uid = None if not tag and hash_key else uid
        if obj:
            function = obj.struct
            # If connected to a tag add it to metadata
            if tag_function_uid:
                function["metadata"]["tag"] = computed_tag
            return mlrun.common.formatters.FunctionFormat.format_obj(function, format_)
        else:
            function_uri = generate_object_uri(project, name, tag, hash_key)
            raise mlrun.errors.MLRunNotFoundError(f"Function not found {function_uri}")

    def _get_function_db_object(
        self,
        session,
        name: str = None,
        project: str = None,
        tag: str = None,
        hash_key: str = None,
    ):
        query = self._query(session, Function, name=name, project=project)
        uid = self._get_function_uid(
            session=session,
            name=name,
            tag=tag,
            hash_key=hash_key,
            project=project,
        )
        if uid:
            query = query.filter(Function.uid == uid)
        return query.one_or_none(), uid

    def _get_function_uid(
        self, session, name: str, tag: str, hash_key: str, project: str
    ):
        computed_tag = tag or "latest"
        if not tag and hash_key:
            return hash_key
        else:
            tag_function_uid = self._resolve_class_tag_uid(
                session, Function, project, name, computed_tag
            )
            if tag_function_uid is None:
                function_uri = generate_object_uri(project, name, tag)
                raise mlrun.errors.MLRunNotFoundError(
                    f"Function tag not found {function_uri}"
                )
            return tag_function_uid

    def _delete_project_functions(self, session: Session, project: str):
        function_names = self._list_project_function_names(session, project)
        self.delete_functions(
            session,
            project,
            function_names,
        )

    def _list_project_function_names(self, session: Session, project: str) -> list[str]:
        return [
            name
            for (name,) in self._query(
                session, distinct(Function.name), project=project
            ).all()
        ]

    def _delete_resources_tags(self, session: Session, project: str):
        for tagged_class in _tagged:
            self._delete(session, tagged_class, project=project)

    def _delete_resources_labels(self, session: Session, project: str):
        for labeled_class in _labeled:
            if hasattr(labeled_class, "project"):
                self._delete(session, labeled_class, project=project)

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

    # ---- Schedules ----
    @retry_on_conflict
    def store_schedule(
        self,
        session: Session,
        project: str,
        name: str,
        kind: mlrun.common.schemas.ScheduleKinds = None,
        scheduled_object: Any = None,
        cron_trigger: mlrun.common.schemas.ScheduleCronTrigger = None,
        labels: dict = None,
        last_run_uri: str = None,
        concurrency_limit: int = None,
        next_run_time: datetime = None,
    ) -> tuple[mlrun.common.schemas.ScheduleRecord, bool]:
        schedule = self._get_schedule_record(
            session=session, project=project, name=name, raise_on_not_found=False
        )
        is_update = schedule is not None

        if not is_update:
            schedule = self._create_schedule_db_record(
                project=project,
                name=name,
                kind=kind,
                scheduled_object=scheduled_object,
                cron_trigger=cron_trigger,
                concurrency_limit=concurrency_limit,
                labels=labels,
                next_run_time=next_run_time,
            )

        self._update_schedule_body(
            schedule=schedule,
            scheduled_object=scheduled_object,
            cron_trigger=cron_trigger,
            labels=labels,
            last_run_uri=last_run_uri,
            concurrency_limit=concurrency_limit,
            next_run_time=next_run_time,
        )

        logger.debug(
            "Storing schedule to db",
            project=schedule.project,
            name=schedule.name,
            kind=schedule.kind,
            cron_trigger=schedule.cron_trigger,
            labels=schedule.labels,
            concurrency_limit=schedule.concurrency_limit,
            scheduled_object=schedule.scheduled_object,
        )

        self._upsert(session, [schedule])

        schedule = self._transform_schedule_record_to_scheme(schedule)
        return schedule, is_update

    def create_schedule(
        self,
        session: Session,
        project: str,
        name: str,
        kind: mlrun.common.schemas.ScheduleKinds,
        scheduled_object: Any,
        cron_trigger: mlrun.common.schemas.ScheduleCronTrigger,
        concurrency_limit: int,
        labels: dict = None,
        next_run_time: datetime = None,
    ) -> mlrun.common.schemas.ScheduleRecord:
        schedule_record = self._create_schedule_db_record(
            project=project,
            name=name,
            kind=kind,
            scheduled_object=scheduled_object,
            cron_trigger=cron_trigger,
            concurrency_limit=concurrency_limit,
            labels=labels,
            next_run_time=next_run_time,
        )

        logger.debug(
            "Saving schedule to db",
            project=schedule_record.project,
            name=schedule_record.name,
            kind=schedule_record.kind,
            cron_trigger=schedule_record.cron_trigger,
            concurrency_limit=schedule_record.concurrency_limit,
            next_run_time=schedule_record.next_run_time,
        )
        self._upsert(session, [schedule_record])

        schedule = self._transform_schedule_record_to_scheme(schedule_record)
        return schedule

    @staticmethod
    def _create_schedule_db_record(
        project: str,
        name: str,
        kind: mlrun.common.schemas.ScheduleKinds,
        scheduled_object: Any,
        cron_trigger: mlrun.common.schemas.ScheduleCronTrigger,
        concurrency_limit: int,
        labels: dict = None,
        next_run_time: datetime = None,
    ) -> Schedule:
        if concurrency_limit is None:
            concurrency_limit = config.httpdb.scheduling.default_concurrency_limit
        if next_run_time is not None:
            # We receive the next_run_time with localized timezone info (e.g +03:00). All the timestamps should be
            # saved in the DB in UTC timezone, therefore we transform next_run_time to UTC as well.
            next_run_time = next_run_time.astimezone(pytz.utc)

        schedule = Schedule(
            project=project,
            name=name,
            kind=kind.value,
            creation_time=datetime.now(timezone.utc),
            concurrency_limit=concurrency_limit,
            next_run_time=next_run_time,
            # these are properties of the object that map manually (using getters and setters) to other column of the
            # table and therefore Pycharm yells that they're unexpected
            scheduled_object=scheduled_object,
            cron_trigger=cron_trigger,
        )

        update_labels(schedule, labels or {})
        return schedule

    def update_schedule(
        self,
        session: Session,
        project: str,
        name: str,
        scheduled_object: Any = None,
        cron_trigger: mlrun.common.schemas.ScheduleCronTrigger = None,
        labels: dict = None,
        last_run_uri: str = None,
        concurrency_limit: int = None,
        next_run_time: datetime = None,
    ):
        schedule = self._get_schedule_record(session, project, name)

        self._update_schedule_body(
            schedule=schedule,
            scheduled_object=scheduled_object,
            cron_trigger=cron_trigger,
            labels=labels,
            last_run_uri=last_run_uri,
            concurrency_limit=concurrency_limit,
            next_run_time=next_run_time,
        )

        logger.debug(
            "Updating schedule in db",
            project=project,
            name=name,
            cron_trigger=cron_trigger,
            labels=labels,
            concurrency_limit=concurrency_limit,
            next_run_time=next_run_time,
        )
        self._upsert(session, [schedule])

    @staticmethod
    def _update_schedule_body(
        schedule: Schedule,
        scheduled_object: Any = None,
        cron_trigger: mlrun.common.schemas.ScheduleCronTrigger = None,
        labels: dict = None,
        last_run_uri: str = None,
        concurrency_limit: int = None,
        next_run_time: datetime = None,
    ):
        # explicitly ensure the updated fields are not None, as they can be empty strings/dictionaries etc.
        if scheduled_object is not None:
            schedule.scheduled_object = scheduled_object

        if cron_trigger is not None:
            schedule.cron_trigger = cron_trigger

        if labels is not None:
            update_labels(schedule, labels)

        if last_run_uri is not None:
            schedule.last_run_uri = last_run_uri

        if concurrency_limit is not None:
            schedule.concurrency_limit = concurrency_limit

        if next_run_time is not None:
            # We receive the next_run_time with localized timezone info (e.g +03:00). All the timestamps should be
            # saved in the DB in UTC timezone, therefore we transform next_run_time to UTC as well.
            schedule.next_run_time = next_run_time.astimezone(pytz.utc)

    def list_schedules(
        self,
        session: Session,
        project: str = None,
        name: str = None,
        labels: list[str] = None,
        kind: mlrun.common.schemas.ScheduleKinds = None,
        as_records: bool = False,
    ) -> list[mlrun.common.schemas.ScheduleRecord]:
        logger.debug("Getting schedules from db", project=project, name=name, kind=kind)
        query = self._query(session, Schedule, kind=kind)
        if project and project != "*":
            query = query.filter(Schedule.project == project)
        if name is not None:
            query = query.filter(generate_query_predicate_for_name(Schedule.name, name))
        labels = label_set(labels)
        query = self._add_labels_filter(session, query, Schedule, labels)

        if as_records:
            return query

        schedules = [
            self._transform_schedule_record_to_scheme(db_schedule)
            for db_schedule in query
        ]
        return schedules

    def get_schedule(
        self, session: Session, project: str, name: str, raise_on_not_found: bool = True
    ) -> typing.Optional[mlrun.common.schemas.ScheduleRecord]:
        logger.debug("Getting schedule from db", project=project, name=name)
        schedule_record = self._get_schedule_record(
            session, project, name, raise_on_not_found
        )
        if not schedule_record:
            return
        schedule = self._transform_schedule_record_to_scheme(schedule_record)
        return schedule

    def delete_schedule(self, session: Session, project: str, name: str):
        logger.debug("Removing schedule from db", project=project, name=name)
        self._delete_class_labels(
            session, Schedule, project=project, name=name, commit=False
        )
        self._delete(session, Schedule, project=project, name=name)

    def delete_project_schedules(self, session: Session, project: str):
        logger.debug("Removing schedules from db", project=project)
        function_names = [
            schedule.name for schedule in self.list_schedules(session, project=project)
        ]
        self.delete_schedules(
            session,
            project,
            names=function_names,
        )

    def delete_schedules(
        self, session: Session, project: str, names: typing.Union[str, list[str]]
    ) -> None:
        logger.debug("Removing schedules from db", project=project, name=names)
        self._delete_multi_objects(
            session=session,
            main_table=Schedule,
            related_tables=[Schedule.Label],
            project=project,
            main_table_identifier=Schedule.name,
            main_table_identifier_values=names,
        )

    def align_schedule_labels(self, session: Session):
        schedules_update = []
        for db_schedule in self.list_schedules(session=session, as_records=True):
            schedule_record = self._transform_schedule_record_to_scheme(db_schedule)
            db_schedule_labels = {
                label.name: label.value for label in db_schedule.labels
            }
            merged_labels = (
                server.api.utils.helpers.merge_schedule_and_schedule_object_labels(
                    labels=db_schedule_labels,
                    scheduled_object=schedule_record.scheduled_object,
                )
            )
            self._update_schedule_body(
                schedule=db_schedule,
                scheduled_object=schedule_record.scheduled_object,
                labels=merged_labels,
            )
            schedules_update.append(db_schedule)
        self._upsert(session, schedules_update)

    @staticmethod
    def _delete_multi_objects(
        session: Session,
        main_table: mlrun.utils.db.BaseModel,
        related_tables: list[mlrun.utils.db.BaseModel],
        project: str,
        main_table_identifier: str,
        main_table_identifier_values: typing.Union[str, list[str]] = None,
    ) -> int:
        """
        Delete multiple objects from the DB, including related tables.
        :param session: SQLAlchemy session.
        :param main_table: The main table to delete from.
        :param related_tables: Related tables to delete from, will be joined with the main table by the identifiers
            since in SQLite the deletion is not always cascading.
        :param project: The project to delete from.
        :param main_table_identifier: The main table attribute to filter by.
        :param main_table_identifier_values: The values corresponding to main_table_identifier to filter by.

        :return: The amount of deleted rows from the main table.
        """
        if not main_table_identifier_values:
            logger.debug(
                "No identifier values provided, skipping deletion",
                project=project,
                tables=[main_table] + related_tables,
            )
            return 0
        for cls in related_tables:
            logger.debug(
                "Removing objects",
                cls=cls,
                project=project,
                main_table_identifier=main_table_identifier,
                main_table_identifier_values_count=len(main_table_identifier_values),
            )

            # The select is mandatory for sqlalchemy 1.4 because
            # query.delete does not support multiple-table criteria within DELETE
            if project != "*":
                subquery = (
                    select(cls.id)
                    .join(main_table)
                    .where(
                        and_(
                            main_table.project == project,
                            or_(
                                main_table_identifier == value
                                for value in main_table_identifier_values
                            ),
                        )
                    )
                    .subquery()
                )
            else:
                subquery = (
                    select(cls.id)
                    .join(main_table)
                    .where(
                        or_(
                            main_table_identifier == value
                            for value in main_table_identifier_values
                        )
                    )
                    .subquery()
                )
            stmt = (
                delete(cls)
                .where(cls.id.in_(aliased(subquery)))
                .execution_options(synchronize_session=False)
            )

            # Execute the delete statement
            execution_obj = session.execute(stmt)
            logger.debug(
                "Removed rows from related table",
                rowcount=execution_obj.rowcount,
                cls=cls,
                main_table=main_table,
                project=project,
            )
        if project != "*":
            query = session.query(main_table).filter(
                and_(
                    main_table.project == project,
                    or_(
                        main_table_identifier == value
                        for value in main_table_identifier_values
                    ),
                )
            )
        else:
            query = session.query(main_table).filter(
                or_(
                    main_table_identifier == value
                    for value in main_table_identifier_values
                ),
            )

        deletions_count = query.delete(synchronize_session=False)
        log_kwargs = {
            "deletions_count": deletions_count,
            "main_table": main_table,
            "project": project,
            "main_table_identifier": main_table_identifier,
        }
        logger.debug("Removed rows from table", **log_kwargs)
        session.commit()
        return deletions_count

    def _get_schedule_record(
        self, session: Session, project: str, name: str, raise_on_not_found: bool = True
    ) -> Schedule:
        query = self._query(session, Schedule, project=project, name=name)
        schedule_record = query.one_or_none()
        if not schedule_record and raise_on_not_found:
            raise mlrun.errors.MLRunNotFoundError(
                f"Schedule not found: project={project}, name={name}"
            )
        return schedule_record

    def _delete_feature_vectors(self, session: Session, project: str):
        logger.debug("Removing feature-vectors from db", project=project)
        for feature_vector_name in self._list_project_feature_vector_names(
            session, project
        ):
            self.delete_feature_vector(session, project, feature_vector_name)

    def _list_project_feature_vector_names(
        self, session: Session, project: str
    ) -> list[str]:
        return [
            name
            for (name,) in self._query(
                session, distinct(FeatureVector.name), project=project
            ).all()
        ]

    def tag_objects_v2(
        self,
        session,
        objs,
        project: str,
        name: str,
        obj_name_attribute: str = "name",
    ):
        tags = []
        for obj in objs:
            query = self._query(
                session,
                obj.Tag,
                name=name,
                project=project,
                obj_name=getattr(obj, obj_name_attribute),
            )

            tag = query.one_or_none()
            if not tag:
                tag = obj.Tag(
                    project=project,
                    name=name,
                    obj_name=getattr(obj, obj_name_attribute),
                )
            tag.obj_id = obj.id
            tags.append(tag)
        self._upsert(session, tags)

    # ---- Projects ----
    def create_project(self, session: Session, project: mlrun.common.schemas.Project):
        logger.debug("Creating project in DB", project_name=project.metadata.name)
        created = datetime.utcnow()
        project.metadata.created = created
        # TODO: handle taking out the functions/workflows/artifacts out of the project and save them separately
        project_record = Project(
            name=project.metadata.name,
            description=project.spec.description,
            source=project.spec.source,
            state=project.status.state,
            created=created,
            owner=project.spec.owner,
            default_function_node_selector=project.spec.default_function_node_selector,
            full_object=project.dict(),
        )
        labels = project.metadata.labels or {}
        update_labels(project_record, labels)

        objects_to_store = [project_record]
        self._append_project_summary(project, objects_to_store)
        self._upsert(session, objects_to_store)

    @staticmethod
    def _append_project_summary(project, objects_to_store):
        summary = mlrun.common.schemas.ProjectSummary(
            name=project.metadata.name,
        )
        project_summary = ProjectSummary(
            project=project.metadata.name,
            summary=summary.dict(),
            updated=datetime.now(timezone.utc),
        )
        objects_to_store.append(project_summary)

    @retry_on_conflict
    def store_project(
        self, session: Session, name: str, project: mlrun.common.schemas.Project
    ):
        logger.debug(
            "Storing project in DB",
            name=name,
            project_metadata=project.metadata,
            project_owner=project.spec.owner,
            project_desired_state=project.spec.desired_state,
            default_function_node_selector=project.spec.default_function_node_selector,
            project_status=project.status,
        )
        self._normalize_project_parameters(project)

        project_record = self._get_project_record(
            session, name, raise_on_not_found=False
        )
        if not project_record:
            self.create_project(session, project)
        else:
            self._update_project_record_from_project(session, project_record, project)

    @staticmethod
    def _normalize_project_parameters(project: mlrun.common.schemas.Project):
        # remove leading & trailing whitespaces from the project parameters keys and values to prevent duplications
        if project.spec.params:
            project.spec.params = {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in project.spec.params.items()
            }

    def patch_project(
        self,
        session: Session,
        name: str,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ):
        logger.debug("Patching project in DB", name=name, patch_mode=patch_mode)
        project_record = self._get_project_record(session, name)
        self._patch_project_record_from_project(
            session, name, project_record, project, patch_mode
        )

    def get_project(
        self,
        session: Session,
        name: str = None,
        project_id: int = None,
    ) -> mlrun.common.schemas.ProjectOut:
        project_record = self._get_project_record(session, name, project_id)

        return self._transform_project_record_to_schema(session, project_record)

    def delete_project(
        self,
        session: Session,
        name: str,
        deletion_strategy: mlrun.common.schemas.DeletionStrategy = mlrun.common.schemas.DeletionStrategy.default(),
    ):
        logger.debug(
            "Deleting project from DB", name=name, deletion_strategy=deletion_strategy
        )
        self._delete_project_summary(session, name)
        self._delete(session, Project, name=name)

    def list_projects(
        self,
        session: Session,
        owner: str = None,
        format_: mlrun.common.formatters.ProjectFormat = mlrun.common.formatters.ProjectFormat.full,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: typing.Optional[list[str]] = None,
    ) -> mlrun.common.schemas.ProjectsOutput:
        query = self._query(session, Project, owner=owner, state=state)

        # if format is name_only, we don't need to query the full project object, we can just query the name
        # and return it as a list of strings
        if format_ == mlrun.common.formatters.ProjectFormat.name_only:
            query = self._query(session, Project.name, owner=owner, state=state)

        # attach filters to the query
        if labels:
            query = self._add_labels_filter(session, query, Project, labels)
        if names is not None:
            query = query.filter(Project.name.in_(names))

        project_records = query.all()

        # format the projects according to the requested format
        projects = []
        for project_record in project_records:
            if format_ == mlrun.common.formatters.ProjectFormat.name_only:
                # can't use formatter as we haven't queried the entire object anyway
                projects.append(project_record.name)
            else:
                projects.append(
                    mlrun.common.formatters.ProjectFormat.format_obj(
                        self._transform_project_record_to_schema(
                            session, project_record
                        ),
                        format_,
                    )
                )
        return mlrun.common.schemas.ProjectsOutput(projects=projects)

    def get_project_summary(
        self,
        session,
        project: str,
    ) -> typing.Optional[mlrun.common.schemas.ProjectSummary]:
        project_summary_record = self._query(
            session,
            ProjectSummary,
            project=project,
        ).one_or_none()
        if not project_summary_record:
            raise mlrun.errors.MLRunNotFoundError(
                f"Project summary not found: {project=}"
            )

        project_summary_record.summary["name"] = project_summary_record.project
        project_summary_record.summary["updated"] = project_summary_record.updated
        return mlrun.common.schemas.ProjectSummary(**project_summary_record.summary)

    def list_project_summaries(
        self,
        session: Session,
        owner: str = None,
        labels: list[str] = None,
        state: mlrun.common.schemas.ProjectState = None,
        names: list[str] = None,
    ):
        project_query = self._query(session, Project.name)
        if owner:
            project_query = project_query.filter(Project.owner == owner)
        if state:
            project_query = project_query.filter(Project.state == state)
        if labels:
            project_query = self._add_labels_filter(
                session, project_query, Project, labels
            )
        if names:
            project_query = project_query.filter(Project.name.in_(names))

        project_subquery = project_query.subquery()
        project_alias = aliased(Project, project_subquery)

        query = self._query(session, ProjectSummary)
        query = query.join(project_alias, ProjectSummary.project == project_alias.name)

        project_summaries = query.all()
        project_summaries_results = []
        for project_summary in project_summaries:
            project_summary.summary["updated"] = project_summary.updated
            project_summaries_results.append(
                mlrun.common.schemas.ProjectSummary(**project_summary.summary)
            )

        return project_summaries_results

    def refresh_project_summaries(
        self,
        session: Session,
        project_summaries: list[mlrun.common.schemas.ProjectSummary],
    ):
        """
        This method updates the summaries of projects that have associated projects in the database
        and removes project summaries that no longer have associated projects.
        """

        summary_dicts = {summary.name: summary.dict() for summary in project_summaries}

        # Create a query for project summaries with associated projects
        existing_summaries_query = (
            session.query(ProjectSummary)
            .outerjoin(Project, Project.name == ProjectSummary.project)
            .filter(ProjectSummary.project.in_(summary_dicts.keys()))
        )

        associated_summaries = existing_summaries_query.filter(
            Project.id.is_not(None)
        ).all()

        orphaned_summaries = existing_summaries_query.filter(Project.id.is_(None)).all()

        # Update the summaries of projects that have associated projects
        for project_summary in associated_summaries:
            project_summary.summary = summary_dicts.get(project_summary.project)
            project_summary.updated = datetime.now(timezone.utc)
            session.add(project_summary)

        # To avoid race conditions where a project might be deleted after its summary is queried
        # but before the transaction completes, we delete project summaries that do not have
        # any associated projects.
        if orphaned_summaries:
            projects_names = [summary.project for summary in orphaned_summaries]
            logger.debug(
                "Deleting project summaries that do not have associated projects",
                projects=projects_names,
            )

            for summary in orphaned_summaries:
                session.delete(summary)

        self._commit(session, associated_summaries + orphaned_summaries)

    def _delete_project_summary(
        self,
        session: Session,
        name: str,
    ):
        logger.debug("Deleting project summary from DB", name=name)
        self._delete(session, ProjectSummary, project=name)

    async def get_project_resources_counters(
        self,
    ) -> tuple[
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
    ]:
        results = await asyncio.gather(
            fastapi.concurrency.run_in_threadpool(
                server.api.db.session.run_function_with_new_db_session,
                self._calculate_files_counters,
            ),
            fastapi.concurrency.run_in_threadpool(
                server.api.db.session.run_function_with_new_db_session,
                self._calculate_schedules_counters,
            ),
            fastapi.concurrency.run_in_threadpool(
                server.api.db.session.run_function_with_new_db_session,
                self._calculate_feature_sets_counters,
            ),
            fastapi.concurrency.run_in_threadpool(
                server.api.db.session.run_function_with_new_db_session,
                self._calculate_models_counters,
            ),
            fastapi.concurrency.run_in_threadpool(
                server.api.db.session.run_function_with_new_db_session,
                self._calculate_runs_counters,
            ),
        )
        (
            project_to_files_count,
            (
                project_to_schedule_count,
                project_to_schedule_pending_jobs_count,
                project_to_schedule_pending_workflows_count,
            ),
            project_to_feature_set_count,
            project_to_models_count,
            (
                project_to_recent_completed_runs_count,
                project_to_recent_failed_runs_count,
                project_to_running_runs_count,
            ),
        ) = results
        return (
            project_to_files_count,
            project_to_schedule_count,
            project_to_schedule_pending_jobs_count,
            project_to_schedule_pending_workflows_count,
            project_to_feature_set_count,
            project_to_models_count,
            project_to_recent_completed_runs_count,
            project_to_recent_failed_runs_count,
            project_to_running_runs_count,
        )

    @staticmethod
    def _calculate_functions_counters(session) -> dict[str, int]:
        functions_count_per_project = (
            session.query(Function.project, func.count(distinct(Function.name)))
            .group_by(Function.project)
            .all()
        )
        project_to_function_count = {
            result[0]: result[1] for result in functions_count_per_project
        }
        return project_to_function_count

    @staticmethod
    def _calculate_schedules_counters(
        session,
    ) -> [dict[str, int], dict[str, int], dict[str, int]]:
        schedules_count_per_project = (
            session.query(Schedule.project, func.count(distinct(Schedule.name)))
            .group_by(Schedule.project)
            .all()
        )
        project_to_schedule_count = {
            result[0]: result[1] for result in schedules_count_per_project
        }

        next_day = datetime.now(timezone.utc) + timedelta(hours=24)

        schedules_pending_count_per_project = (
            session.query(
                Schedule.project,
                Schedule.name,
                # The logic here is the following:
                # If the schedule has a label with the name "workflow" then we take the value of that label
                # name. Otherwise, we take the value of the label with the name "kind".
                # The reason for that is that for schedule workflow we have both workflow label and kind label
                # with job, and on schedule job we have only kind label with job and because of that we first
                # want to check for workflow label and if it doesn't exist then we take the kind label.
                func.coalesce(
                    func.max(
                        case(
                            [
                                (
                                    Schedule.Label.name
                                    == mlrun_constants.MLRunInternalLabels.workflow,
                                    Schedule.Label.name,
                                )
                            ],
                            else_=None,
                        )
                    ),
                    func.max(
                        case(
                            [
                                (
                                    Schedule.Label.name
                                    == mlrun_constants.MLRunInternalLabels.kind,
                                    Schedule.Label.value,
                                )
                            ],
                            else_=None,
                        )
                    ),
                ).label("preferred_label_value"),
            )
            .join(Schedule.Label, Schedule.Label.parent == Schedule.id)
            .filter(Schedule.next_run_time < next_day)
            .filter(Schedule.next_run_time >= datetime.now(timezone.utc))
            .filter(
                Schedule.Label.name.in_(
                    [
                        mlrun_constants.MLRunInternalLabels.workflow,
                        mlrun_constants.MLRunInternalLabels.kind,
                    ]
                )
            )
            .group_by(Schedule.project, Schedule.name)
            .all()
        )

        project_to_schedule_pending_jobs_count = collections.defaultdict(int)
        project_to_schedule_pending_workflows_count = collections.defaultdict(int)

        for result in schedules_pending_count_per_project:
            project_name, schedule_name, kind = result
            if kind == mlrun_constants.MLRunInternalLabels.workflow:
                project_to_schedule_pending_workflows_count[project_name] += 1
            elif kind == mlrun.common.schemas.ScheduleKinds.job:
                project_to_schedule_pending_jobs_count[project_name] += 1

        return (
            project_to_schedule_count,
            project_to_schedule_pending_jobs_count,
            project_to_schedule_pending_workflows_count,
        )

    @staticmethod
    def _calculate_feature_sets_counters(session) -> dict[str, int]:
        feature_sets_count_per_project = (
            session.query(FeatureSet.project, func.count(distinct(FeatureSet.name)))
            .group_by(FeatureSet.project)
            .all()
        )
        project_to_feature_set_count = {
            result[0]: result[1] for result in feature_sets_count_per_project
        }
        return project_to_feature_set_count

    def _calculate_models_counters(self, session) -> dict[str, int]:
        # We're using the "most_recent" which gives us only one version of each artifact key, which is what we want to
        # count (artifact count, not artifact versions count)
        model_artifacts = self._find_artifacts(
            session,
            None,
            kind=mlrun.common.schemas.ArtifactCategories.model,
            most_recent=True,
        )
        project_to_models_count = collections.defaultdict(int)
        for model_artifact in model_artifacts:
            project_to_models_count[model_artifact.project] += 1
        return project_to_models_count

    def _calculate_files_counters(self, session) -> dict[str, int]:
        # We're using the "most_recent" flag which gives us only one version of each artifact key, which is what we
        # want to count (artifact count, not artifact versions count)
        file_artifacts = self._find_artifacts(
            session,
            None,
            category=mlrun.common.schemas.ArtifactCategories.other,
            most_recent=True,
        )
        project_to_files_count = collections.defaultdict(int)
        for file_artifact in file_artifacts:
            project_to_files_count[file_artifact.project] += 1
        return project_to_files_count

    @staticmethod
    def _calculate_runs_counters(
        session,
    ) -> tuple[
        dict[str, int],
        dict[str, int],
        dict[str, int],
    ]:
        running_runs_count_per_project = (
            session.query(Run.project, func.count(distinct(Run.name)))
            .filter(
                Run.state.in_(
                    mlrun.common.runtimes.constants.RunStates.non_terminal_states()
                )
            )
            .group_by(Run.project)
            .all()
        )
        project_to_running_runs_count = {
            result[0]: result[1] for result in running_runs_count_per_project
        }

        one_day_ago = datetime.now() - timedelta(hours=24)
        recent_failed_runs_count_per_project = (
            session.query(Run.project, func.count(distinct(Run.name)))
            .filter(
                Run.state.in_(
                    [
                        mlrun.common.runtimes.constants.RunStates.error,
                        mlrun.common.runtimes.constants.RunStates.aborted,
                    ]
                )
            )
            .filter(Run.start_time >= one_day_ago)
            .group_by(Run.project)
            .all()
        )
        project_to_recent_failed_runs_count = {
            result[0]: result[1] for result in recent_failed_runs_count_per_project
        }

        recent_completed_runs_count_per_project = (
            session.query(Run.project, func.count(distinct(Run.name)))
            .filter(
                Run.state.in_(
                    [
                        mlrun.common.runtimes.constants.RunStates.completed,
                    ]
                )
            )
            .filter(Run.start_time >= one_day_ago)
            .group_by(Run.project)
            .all()
        )
        project_to_recent_completed_runs_count = {
            result[0]: result[1] for result in recent_completed_runs_count_per_project
        }
        return (
            project_to_recent_completed_runs_count,
            project_to_recent_failed_runs_count,
            project_to_running_runs_count,
        )

    def _update_project_record_from_project(
        self,
        session: Session,
        project_record: Project,
        project: mlrun.common.schemas.Project,
    ):
        project.metadata.created = project_record.created
        project_dict = project.dict()
        # TODO: handle taking out the functions/workflows/artifacts out of the project and save them separately
        project_record.full_object = project_dict
        project_record.description = project.spec.description
        project_record.source = project.spec.source
        project_record.owner = project.spec.owner
        project_record.state = project.status.state
        project_record.default_function_node_selector = (
            project.spec.default_function_node_selector
        )
        labels = project.metadata.labels or {}
        update_labels(project_record, labels)
        self._upsert(session, [project_record])

    def _patch_project_record_from_project(
        self,
        session: Session,
        name: str,
        project_record: Project,
        project: dict,
        patch_mode: mlrun.common.schemas.PatchMode,
    ):
        project.setdefault("metadata", {})["created"] = project_record.created
        strategy = patch_mode.to_mergedeep_strategy()
        project_record_full_object = project_record.full_object
        mergedeep.merge(project_record_full_object, project, strategy=strategy)

        # If a bad kind value was passed, it will fail here (return 422 to caller)
        project = mlrun.common.schemas.Project(**project_record_full_object)
        self.store_project(
            session,
            name,
            project,
        )

        project_record.full_object = project_record_full_object
        self._upsert(session, [project_record])

    def is_project_exists(self, session: Session, name: str):
        project_record = self._get_project_record(
            session, name, raise_on_not_found=False
        )
        if not project_record:
            return False
        return True

    def _get_project_record(
        self,
        session: Session,
        name: str = None,
        project_id: int = None,
        raise_on_not_found: bool = True,
    ) -> typing.Optional[Project]:
        if not any([project_id, name]):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "One of 'name' or 'project_id' must be provided"
            )
        project_record = self._query(
            session, Project, name=name, id=project_id
        ).one_or_none()
        if not project_record:
            if not raise_on_not_found:
                return None
            raise mlrun.errors.MLRunNotFoundError(
                f"Project not found: name={name}, project_id={project_id}"
            )

        return project_record

    def verify_project_has_no_related_resources(self, session: Session, name: str):
        artifacts = self._find_artifacts(session, name, "*")
        self._verify_empty_list_of_project_related_resources(
            name, artifacts, "artifacts"
        )
        logs = self._list_logs(session, name)
        self._verify_empty_list_of_project_related_resources(name, logs, "logs")
        runs = self._find_runs(session, None, name, []).all()
        self._verify_empty_list_of_project_related_resources(name, runs, "runs")
        notifications = []
        for cls in _with_notifications:
            notifications.extend(self._get_db_notifications(session, cls, project=name))
        self._verify_empty_list_of_project_related_resources(
            name, notifications, "notifications"
        )
        schedules = self.list_schedules(session, project=name)
        self._verify_empty_list_of_project_related_resources(
            name, schedules, "schedules"
        )
        functions = self._list_project_function_names(session, name)
        self._verify_empty_list_of_project_related_resources(
            name, functions, "functions"
        )
        feature_sets = self._list_project_feature_set_names(session, name)
        self._verify_empty_list_of_project_related_resources(
            name, feature_sets, "feature_sets"
        )
        feature_vectors = self._list_project_feature_vector_names(session, name)
        self._verify_empty_list_of_project_related_resources(
            name, feature_vectors, "feature_vectors"
        )

    def delete_project_related_resources(self, session: Session, name: str):
        self.del_artifacts(session, project=name)
        self._delete_logs(session, name)
        self.delete_run_notifications(session, project=name)
        self.delete_alert_notifications(session, project=name)
        self.del_runs(session, project=name)
        self.delete_project_schedules(session, name)
        self._delete_project_functions(session, name)
        self._delete_feature_sets(session, name)
        self._delete_feature_vectors(session, name)
        self._delete_background_tasks(session, project=name)
        self.delete_datastore_profiles(session, project=name)

        # resources deletion should remove their tags and labels as well, but doing another try in case there are
        # orphan resources
        self._delete_resources_tags(session, name)
        self._delete_resources_labels(session, name)

    @staticmethod
    def _verify_empty_list_of_project_related_resources(
        project: str, resources: list, resource_name: str
    ):
        if resources:
            raise mlrun.errors.MLRunPreconditionFailedError(
                f"Project {project} can not be deleted since related resources found: {resource_name}"
            )

    def _get_record_by_name_tag_and_uid(
        self,
        session,
        cls,
        project: str,
        name: str,
        tag: str = None,
        uid: str = None,
        obj_name_attribute="name",
    ):
        kwargs = {obj_name_attribute: name, "project": project}
        query = self._query(session, cls, **kwargs)
        computed_tag = tag or "latest"
        object_tag_uid = None
        if tag or not uid:
            object_tag_uid = self._resolve_class_tag_uid(
                session, cls, project, name, computed_tag
            )
            if object_tag_uid is None:
                return None, None, None
            uid = object_tag_uid
        if uid:
            query = query.filter(cls.uid == uid)
        return computed_tag, object_tag_uid, query.one_or_none()

    # ---- Feature sets ----
    def create_feature_set(
        self,
        session,
        project,
        feature_set: mlrun.common.schemas.FeatureSet,
        versioned=True,
    ) -> str:
        (
            uid,
            tag,
            feature_set_dict,
        ) = self._validate_and_enrich_record_for_creation(
            session, feature_set, FeatureSet, project, versioned
        )

        db_feature_set = FeatureSet(project=project)
        self._update_db_record_from_object_dict(db_feature_set, feature_set_dict, uid)
        self._update_feature_set_spec(db_feature_set, feature_set_dict)

        self._upsert(session, [db_feature_set])
        self.tag_objects_v2(session, [db_feature_set], project, tag)

        return uid

    def patch_feature_set(
        self,
        session,
        project,
        name,
        feature_set_patch: dict,
        tag=None,
        uid=None,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ) -> str:
        feature_set_record = self._get_feature_set(session, project, name, tag, uid)
        if not feature_set_record:
            feature_set_uri = generate_object_uri(project, name, tag)
            raise mlrun.errors.MLRunNotFoundError(
                f"Feature-set not found {feature_set_uri}"
            )

        feature_set_struct = feature_set_record.dict(exclude_none=True)
        # using mergedeep for merging the patch content into the existing dictionary
        strategy = patch_mode.to_mergedeep_strategy()
        mergedeep.merge(feature_set_struct, feature_set_patch, strategy=strategy)

        versioned = feature_set_record.metadata.uid is not None

        # If a bad kind value was passed, it will fail here (return 422 to caller)
        feature_set = mlrun.common.schemas.FeatureSet(**feature_set_struct)
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

    def get_feature_set(
        self,
        session,
        project: str,
        name: str,
        tag: str = None,
        uid: str = None,
    ) -> mlrun.common.schemas.FeatureSet:
        feature_set = self._get_feature_set(session, project, name, tag, uid)
        if not feature_set:
            feature_set_uri = generate_object_uri(project, name, tag)
            raise mlrun.errors.MLRunNotFoundError(
                f"Feature-set not found {feature_set_uri}"
            )

        return feature_set

    def _get_feature_set(
        self,
        session,
        project: str,
        name: str,
        tag: str = None,
        uid: str = None,
    ):
        (
            computed_tag,
            feature_set_tag_uid,
            db_feature_set,
        ) = self._get_record_by_name_tag_and_uid(
            session, FeatureSet, project, name, tag, uid
        )
        if db_feature_set:
            feature_set = self._transform_feature_set_model_to_schema(db_feature_set)

            # If connected to a tag add it to metadata
            if feature_set_tag_uid:
                feature_set.metadata.tag = computed_tag
            return feature_set
        else:
            return None

    def _get_records_to_tags_map(self, session, cls, project, tag, name=None):
        # Find object IDs by tag, project and object-name (which is a like query)
        tag_query = self._query(session, cls.Tag, project=project, name=tag)
        if name:
            tag_query = tag_query.filter(
                generate_query_predicate_for_name(cls.Tag.obj_name, name)
            )

        # Generate a mapping from each object id (note: not uid, it's the DB ID) to its associated tags.
        obj_id_tags = {}
        for row in tag_query:
            if row.obj_id in obj_id_tags:
                obj_id_tags[row.obj_id].append(row.name)
            else:
                obj_id_tags[row.obj_id] = [row.name]
        return obj_id_tags

    def _generate_records_with_tags_assigned(
        self, object_record, transform_fn, obj_id_tags, default_tag=None, format_=None
    ):
        # Using a similar mechanism here to assign tags to feature sets as is used in list_functions. Please refer
        # there for some comments explaining the logic.
        results = []
        if default_tag:
            results.append(transform_fn(object_record, default_tag, format_=format_))
        else:
            object_tags = obj_id_tags.get(object_record.id, [])
            if len(object_tags) == 0 and not object_record.uid.startswith(
                unversioned_tagged_object_uid_prefix
            ):
                new_object = transform_fn(object_record, format_=format_)
                results.append(new_object)
            else:
                for object_tag in object_tags:
                    results.append(
                        transform_fn(object_record, object_tag, format_=format_)
                    )
        return results

    @staticmethod
    def _generate_feature_set_digest(feature_set: mlrun.common.schemas.FeatureSet):
        return mlrun.common.schemas.FeatureSetDigestOutput(
            metadata=feature_set.metadata,
            spec=mlrun.common.schemas.FeatureSetDigestSpec(
                entities=feature_set.spec.entities,
                features=feature_set.spec.features,
            ),
        )

    def _generate_feature_or_entity_list_query(
        self,
        session,
        query_class,
        project: str,
        feature_set_keys,
        name: str = None,
        tag: str = None,
        labels: list[str] = None,
    ):
        # Query the actual objects to be returned
        query = (
            session.query(FeatureSet, query_class)
            .filter_by(project=project)
            .join(query_class)
        )

        if name:
            query = query.filter(
                generate_query_predicate_for_name(query_class.name, name)
            )
        if labels:
            query = self._add_labels_filter(session, query, query_class, labels)
        if tag:
            query = query.filter(FeatureSet.id.in_(feature_set_keys))

        return query

    def list_features(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        entities: list[str] = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.FeaturesOutput:
        # We don't filter by feature-set name here, as the name parameter refers to features
        feature_set_id_tags = self._get_records_to_tags_map(
            session, FeatureSet, project, tag, name=None
        )

        query = self._generate_feature_or_entity_list_query(
            session, Feature, project, feature_set_id_tags.keys(), name, tag, labels
        )

        if entities:
            query = query.join(FeatureSet.entities).filter(Entity.name.in_(entities))

        features_results = []
        transform_feature_set_model_to_schema = MemoizationCache(
            self._transform_feature_set_model_to_schema
        ).memoize
        generate_feature_set_digest = MemoizationCache(
            self._generate_feature_set_digest
        ).memoize

        for row in query:
            feature_record = mlrun.common.schemas.FeatureRecord.from_orm(row.Feature)
            feature_name = feature_record.name

            feature_sets = self._generate_records_with_tags_assigned(
                row.FeatureSet,
                transform_feature_set_model_to_schema,
                feature_set_id_tags,
                tag,
            )

            for feature_set in feature_sets:
                # Get the feature from the feature-set full structure, as it may contain extra fields (which are not
                # in the DB)
                feature = next(
                    (
                        feature
                        for feature in feature_set.spec.features
                        if feature.name == feature_name
                    ),
                    None,
                )
                if not feature:
                    raise mlrun.errors.MLRunInternalServerError(
                        "Inconsistent data in DB - features in DB not in feature-set document"
                    )

                feature_set_digest = generate_feature_set_digest(feature_set)

                features_results.append(
                    mlrun.common.schemas.FeatureListOutput(
                        feature=feature,
                        feature_set_digest=feature_set_digest,
                    )
                )
        return mlrun.common.schemas.FeaturesOutput(features=features_results)

    @staticmethod
    def _dedup_and_append_feature_set(
        feature_set, feature_set_id_to_index, feature_set_digests_v2
    ):
        # dedup feature set list
        # we can rely on the object ID because SQLAlchemy already avoids duplication at the object
        # level, and the conversion from "model" to "schema" retains this property
        feature_set_obj_id = id(feature_set)
        feature_set_index = feature_set_id_to_index.get(feature_set_obj_id, None)
        if feature_set_index is None:
            feature_set_index = len(feature_set_id_to_index)
            feature_set_id_to_index[feature_set_obj_id] = feature_set_index
            feature_set_digests_v2.append(
                FeatureSetDigestOutputV2(
                    feature_set_index=feature_set_index,
                    metadata=feature_set.metadata,
                    spec=FeatureSetDigestSpecV2(
                        entities=feature_set.spec.entities,
                    ),
                )
            )
        return feature_set_index

    @staticmethod
    def _build_feature_mapping_from_feature_set(feature_set):
        result = {}
        for feature in feature_set.spec.features:
            result[feature.name] = feature
        return result

    @staticmethod
    def _build_entity_mapping_from_feature_set(feature_set):
        result = {}
        for entity in feature_set.spec.entities:
            result[entity.name] = entity
        return result

    def list_features_v2(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        entities: list[str] = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.FeaturesOutputV2:
        # We don't filter by feature-set name here, as the name parameter refers to features
        feature_set_id_tags = self._get_records_to_tags_map(
            session, FeatureSet, project, tag, name=None
        )

        query = self._generate_feature_or_entity_list_query(
            session, Feature, project, feature_set_id_tags.keys(), name, tag, labels
        )

        if entities:
            query = query.join(FeatureSet.entities).filter(Entity.name.in_(entities))

        features_with_feature_set_index: list[Feature] = []
        feature_set_digests_v2: list[FeatureSetDigestOutputV2] = []
        feature_set_digest_id_to_index: dict[int, int] = {}

        transform_feature_set_model_to_schema = MemoizationCache(
            self._transform_feature_set_model_to_schema
        ).memoize
        build_feature_mapping_from_feature_set = MemoizationCache(
            self._build_feature_mapping_from_feature_set
        ).memoize

        for row in query:
            feature_record = mlrun.common.schemas.FeatureRecord.from_orm(row.Feature)
            feature_name = feature_record.name

            feature_sets = self._generate_records_with_tags_assigned(
                row.FeatureSet,
                transform_feature_set_model_to_schema,
                feature_set_id_tags,
                tag,
            )

            for feature_set in feature_sets:
                # Get the feature from the feature-set full structure, as it may contain extra fields (which are not
                # in the DB)
                feature_name_to_feature = build_feature_mapping_from_feature_set(
                    feature_set
                )
                feature = feature_name_to_feature.get(feature_name)
                if not feature:
                    raise mlrun.errors.MLRunInternalServerError(
                        "Inconsistent data in DB - features in DB not in feature-set document"
                    )

                feature_set_index = self._dedup_and_append_feature_set(
                    feature_set, feature_set_digest_id_to_index, feature_set_digests_v2
                )
                features_with_feature_set_index.append(
                    feature.copy(update=dict(feature_set_index=feature_set_index))
                )

        return mlrun.common.schemas.FeaturesOutputV2(
            features=features_with_feature_set_index,
            feature_set_digests=feature_set_digests_v2,
        )

    def list_entities(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.EntitiesOutput:
        feature_set_id_tags = self._get_records_to_tags_map(
            session, FeatureSet, project, tag, name=None
        )

        query = self._generate_feature_or_entity_list_query(
            session, Entity, project, feature_set_id_tags.keys(), name, tag, labels
        )

        entities_results = []
        transform_feature_set_model_to_schema = MemoizationCache(
            self._transform_feature_set_model_to_schema
        ).memoize
        generate_feature_set_digest = MemoizationCache(
            self._generate_feature_set_digest
        ).memoize

        for row in query:
            entity_record = mlrun.common.schemas.FeatureRecord.from_orm(row.Entity)
            entity_name = entity_record.name

            feature_sets = self._generate_records_with_tags_assigned(
                row.FeatureSet,
                transform_feature_set_model_to_schema,
                feature_set_id_tags,
                tag,
            )

            for feature_set in feature_sets:
                # Get the feature from the feature-set full structure, as it may contain extra fields (which are not
                # in the DB)
                entity = next(
                    (
                        entity
                        for entity in feature_set.spec.entities
                        if entity.name == entity_name
                    ),
                    None,
                )
                if not entity:
                    raise mlrun.errors.MLRunInternalServerError(
                        "Inconsistent data in DB - entities in DB not in feature-set document"
                    )

                feature_set_digest = generate_feature_set_digest(feature_set)

                entities_results.append(
                    mlrun.common.schemas.EntityListOutput(
                        entity=entity,
                        feature_set_digest=feature_set_digest,
                    )
                )

        return mlrun.common.schemas.EntitiesOutput(entities=entities_results)

    def list_entities_v2(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        labels: list[str] = None,
    ) -> mlrun.common.schemas.EntitiesOutputV2:
        feature_set_id_tags = self._get_records_to_tags_map(
            session, FeatureSet, project, tag, name=None
        )

        query = self._generate_feature_or_entity_list_query(
            session, Entity, project, feature_set_id_tags.keys(), name, tag, labels
        )

        entities_with_feature_set_index: list[Entity] = []
        feature_set_digests_v2: list[FeatureSetDigestOutputV2] = []
        feature_set_digest_id_to_index: dict[int, int] = {}

        transform_feature_set_model_to_schema = MemoizationCache(
            self._transform_feature_set_model_to_schema
        ).memoize
        build_entity_mapping_from_feature_set = MemoizationCache(
            self._build_entity_mapping_from_feature_set
        ).memoize

        for row in query:
            entity_record = mlrun.common.schemas.FeatureRecord.from_orm(row.Entity)
            entity_name = entity_record.name

            feature_sets = self._generate_records_with_tags_assigned(
                row.FeatureSet,
                transform_feature_set_model_to_schema,
                feature_set_id_tags,
                tag,
            )

            for feature_set in feature_sets:
                # Get the feature from the feature-set full structure, as it may contain extra fields (which are not
                # in the DB)
                entity_name_to_feature = build_entity_mapping_from_feature_set(
                    feature_set
                )
                entity = entity_name_to_feature.get(entity_name)
                if not entity:
                    raise mlrun.errors.MLRunInternalServerError(
                        "Inconsistent data in DB - entities in DB not in feature-set document"
                    )

                feature_set_index = self._dedup_and_append_feature_set(
                    feature_set, feature_set_digest_id_to_index, feature_set_digests_v2
                )
                entities_with_feature_set_index.append(
                    entity.copy(update=dict(feature_set_index=feature_set_index))
                )

        return mlrun.common.schemas.EntitiesOutputV2(
            entities=entities_with_feature_set_index,
            feature_set_digests=feature_set_digests_v2,
        )

    @staticmethod
    def _assert_partition_by_parameters(partition_by_enum_cls, partition_by, sort):
        if sort is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "sort parameter must be provided when partition_by is used."
            )
        # For now, name is the only supported value. Remove once more fields are added.
        if partition_by not in partition_by_enum_cls:
            valid_enum_values = [
                enum_value.value for enum_value in partition_by_enum_cls
            ]
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid partition_by given: '{partition_by.value}'. Must be one of {valid_enum_values}"
            )

    @staticmethod
    def _create_partitioned_query(
        session,
        query,
        cls,
        partition_by: typing.Union[
            mlrun.common.schemas.FeatureStorePartitionByField,
            mlrun.common.schemas.RunPartitionByField,
        ],
        rows_per_partition: int,
        partition_sort_by: mlrun.common.schemas.SortField,
        partition_order: mlrun.common.schemas.OrderType,
        max_partitions: int = 0,
    ):
        partition_field = partition_by.to_partition_by_db_field(cls)
        sort_by_field = partition_sort_by.to_db_field(cls)

        row_number_column = (
            func.row_number()
            .over(
                partition_by=partition_field,
                order_by=partition_order.to_order_by_predicate(sort_by_field),
            )
            .label("row_number")
        )

        # Retrieve only the ID from the subquery to minimize the inner table,
        # in the final step we inner join the inner table with the full table.
        query = query.with_entities(cls.id).add_column(row_number_column)
        if max_partitions > 0:
            max_partition_value = (
                func.max(sort_by_field)
                .over(
                    partition_by=partition_field,
                )
                .label("max_partition_value")
            )
            query = query.add_column(max_partition_value)

        # Need to generate a subquery so we can filter based on the row_number, since it
        # is a window function using over().
        subquery = query.subquery()
        if max_partitions == 0:
            result_query = (
                session.query(cls)
                .join(subquery, cls.id == subquery.c.id)
                .filter(subquery.c.row_number <= rows_per_partition)
            )
            return result_query

        result_query = session.query(subquery).filter(
            subquery.c.row_number <= rows_per_partition
        )

        # We query on max-partitions, so need to do another sub-query and order per the latest updated time of
        # a run in the partition.
        partition_rank = (
            func.dense_rank()
            .over(order_by=subquery.c.max_partition_value.desc())
            .label("partition_rank")
        )
        subquery = result_query.add_column(partition_rank).subquery()
        result_query = (
            session.query(cls)
            .join(subquery, cls.id == subquery.c.id)
            .filter(subquery.c.partition_rank <= max_partitions)
        )
        return result_query

    def list_feature_sets(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        state: str = None,
        entities: list[str] = None,
        features: list[str] = None,
        labels: list[str] = None,
        partition_by: mlrun.common.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
        format_: mlrun.common.formatters.FeatureSetFormat = mlrun.common.formatters.FeatureSetFormat.full,
    ) -> mlrun.common.schemas.FeatureSetsOutput:
        obj_id_tags = self._get_records_to_tags_map(
            session, FeatureSet, project, tag, name
        )

        # Query the actual objects to be returned
        query = self._query(session, FeatureSet, project=project, state=state)

        if name is not None:
            query = query.filter(
                generate_query_predicate_for_name(FeatureSet.name, name)
            )
        if tag:
            query = query.filter(FeatureSet.id.in_(obj_id_tags.keys()))
        if entities:
            query = query.join(FeatureSet.entities).filter(Entity.name.in_(entities))
        if features:
            query = query.join(FeatureSet.features).filter(Feature.name.in_(features))
        if labels:
            query = self._add_labels_filter(session, query, FeatureSet, labels)

        if partition_by:
            self._assert_partition_by_parameters(
                mlrun.common.schemas.FeatureStorePartitionByField,
                partition_by,
                partition_sort_by,
            )
            query = self._create_partitioned_query(
                session,
                query,
                FeatureSet,
                partition_by,
                rows_per_partition,
                partition_sort_by,
                partition_order,
            )

        feature_sets = []
        for feature_set_record in query:
            feature_sets.extend(
                self._generate_records_with_tags_assigned(
                    feature_set_record,
                    self._transform_feature_set_model_to_schema,
                    obj_id_tags,
                    tag,
                    format_=format_,
                )
            )
        return mlrun.common.schemas.FeatureSetsOutput(feature_sets=feature_sets)

    def list_feature_sets_tags(
        self,
        session,
        project: str,
    ):
        query = (
            session.query(FeatureSet.name, FeatureSet.Tag.name)
            .filter(FeatureSet.Tag.project == project)
            .join(FeatureSet, FeatureSet.Tag.obj_id == FeatureSet.id)
            .distinct()
        )
        return [(project, row[0], row[1]) for row in query]

    @staticmethod
    def _update_feature_set_features(
        feature_set: FeatureSet, feature_dicts: list[dict]
    ):
        new_features = set(feature_dict["name"] for feature_dict in feature_dicts)
        current_features = set(feature.name for feature in feature_set.features)

        features_to_remove = current_features.difference(new_features)
        features_to_add = new_features.difference(current_features)

        feature_set.features = [
            feature
            for feature in feature_set.features
            if feature.name not in features_to_remove
        ]

        for feature_dict in feature_dicts:
            feature_name = feature_dict["name"]
            if feature_name in features_to_add:
                labels = feature_dict.get("labels") or {}
                feature = Feature(
                    name=feature_dict["name"],
                    value_type=feature_dict["value_type"],
                    labels=[],
                )
                update_labels(feature, labels)
                feature_set.features.append(feature)
            elif feature_name not in features_to_remove:
                # get the existing feature from the feature set
                feature = next(
                    (
                        feature
                        for feature in feature_set.features
                        if feature.name == feature_name
                    ),
                    None,
                )
                if feature:
                    # update it with the new labels in case they were changed
                    labels = feature_dict.get("labels") or {}
                    update_labels(feature, labels)

    @staticmethod
    def _update_feature_set_entities(feature_set: FeatureSet, entity_dicts: list[dict]):
        new_entities = set(entity_dict["name"] for entity_dict in entity_dicts)
        current_entities = set(entity.name for entity in feature_set.entities)

        entities_to_remove = current_entities.difference(new_entities)
        entities_to_add = new_entities.difference(current_entities)

        feature_set.entities = [
            entity
            for entity in feature_set.entities
            if entity.name not in entities_to_remove
        ]

        for entity_dict in entity_dicts:
            if entity_dict["name"] in entities_to_add:
                labels = entity_dict.get("labels") or {}
                entity = Entity(
                    name=entity_dict["name"],
                    value_type=entity_dict["value_type"],
                    labels=[],
                )
                update_labels(entity, labels)
                feature_set.entities.append(entity)

    def _update_feature_set_spec(
        self, feature_set: FeatureSet, new_feature_set_dict: dict
    ):
        feature_set_spec = new_feature_set_dict.get("spec")
        features = feature_set_spec.pop("features", [])
        entities = feature_set_spec.pop("entities", [])
        self._update_feature_set_features(feature_set, features)
        self._update_feature_set_entities(feature_set, entities)

    @staticmethod
    def _common_object_validate_and_perform_uid_change(
        object_dict: dict,
        tag,
        versioned,
        existing_uid=None,
    ):
        uid = fill_object_hash(object_dict, "uid", tag)
        if not versioned:
            uid = f"{unversioned_tagged_object_uid_prefix}{tag}"
            object_dict["metadata"]["uid"] = uid

        # If object was referenced by UID, the request cannot modify it
        if existing_uid and uid != existing_uid:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Changing uid for an object referenced by its uid"
            )
        return uid

    @staticmethod
    def _update_db_record_from_object_dict(
        db_object,
        common_object_dict: dict,
        uid,
    ):
        db_object.name = common_object_dict["metadata"]["name"]
        updated_datetime = datetime.now(timezone.utc)
        db_object.updated = updated_datetime
        if not db_object.created:
            db_object.created = common_object_dict["metadata"].pop(
                "created", None
            ) or datetime.now(timezone.utc)
        db_object.state = common_object_dict.get("status", {}).get("state")
        db_object.uid = uid

        common_object_dict["metadata"]["updated"] = str(updated_datetime)
        common_object_dict["metadata"]["created"] = str(db_object.created)

        # In case of an unversioned object, we don't want to return uid to user queries. However,
        # the uid DB field has to be set, since it's used for uniqueness in the DB.
        if uid.startswith(unversioned_tagged_object_uid_prefix):
            common_object_dict["metadata"].pop("uid", None)

        db_object.full_object = common_object_dict

        # labels are stored in a separate table
        labels = common_object_dict["metadata"].pop("labels", {}) or {}
        update_labels(db_object, labels)

    @retry_on_conflict
    def store_feature_set(
        self,
        session,
        project,
        name,
        feature_set: mlrun.common.schemas.FeatureSet,
        tag=None,
        uid=None,
        versioned=True,
        always_overwrite=False,
    ) -> str:
        return self._store_tagged_object(
            session,
            FeatureSet,
            project,
            name,
            feature_set,
            tag=tag,
            uid=uid,
            versioned=versioned,
            always_overwrite=always_overwrite,
        )

    def _store_tagged_object(
        self,
        session,
        cls,
        project,
        name,
        tagged_object: typing.Union[
            mlrun.common.schemas.FeatureVector,
            mlrun.common.schemas.FeatureSet,
        ],
        tag=None,
        uid=None,
        versioned=True,
        always_overwrite=False,
    ):
        original_uid = uid

        # record with the given tag/uid
        _, _, existing_tagged_object = self._get_record_by_name_tag_and_uid(
            session, cls, project, name, tag, uid
        )

        tagged_object_dict = tagged_object.dict(exclude_none=True)

        # get the computed uid
        uid = self._common_object_validate_and_perform_uid_change(
            tagged_object_dict, tag, versioned, original_uid
        )

        if existing_tagged_object:
            if uid == existing_tagged_object.uid or always_overwrite:
                db_tagged_object = existing_tagged_object
            else:
                # In case an object with the given tag (or 'latest' which is the default) and name, but different uid
                # was found - Check If an object with the same computed uid but different tag already exists
                # and re-tag it.
                if self._re_tag_existing_object(session, cls, project, name, tag, uid):
                    return uid

                db_tagged_object = cls(project=project)

            self._update_db_record_from_object_dict(
                db_tagged_object, tagged_object_dict, uid
            )

            if cls == FeatureSet:
                self._update_feature_set_spec(db_tagged_object, tagged_object_dict)
            self._upsert(session, [db_tagged_object])
            if tag:
                self.tag_objects_v2(session, [db_tagged_object], project, tag)
            return uid

        # Object with the given tag/uid doesn't exist
        # Check if this is a re-tag of existing object - search by uid only
        if self._re_tag_existing_object(session, cls, project, name, tag, uid):
            return uid

        tagged_object.metadata.tag = tag
        return self._create_tagged_object(
            session, project, cls, tagged_object, versioned
        )

    def _create_tagged_object(
        self,
        session,
        project,
        cls,
        tagged_object: typing.Union[
            mlrun.common.schemas.FeatureVector,
            mlrun.common.schemas.FeatureSet,
            dict,
        ],
        versioned=True,
    ):
        uid, tag, tagged_object_dict = self._validate_and_enrich_record_for_creation(
            session, tagged_object, cls, project, versioned
        )

        db_tagged_object = cls(project=project)

        self._update_db_record_from_object_dict(
            db_tagged_object, tagged_object_dict, uid
        )
        if cls == FeatureSet:
            self._update_feature_set_spec(db_tagged_object, tagged_object_dict)

        self._upsert(session, [db_tagged_object])
        self.tag_objects_v2(session, [db_tagged_object], project, tag)

        return uid

    def _re_tag_existing_object(
        self,
        session,
        cls,
        project,
        name,
        tag,
        uid,
        obj_name_attribute: str = "name",
    ):
        _, _, existing_object = self._get_record_by_name_tag_and_uid(
            session,
            cls,
            project,
            name,
            None,
            uid,
            obj_name_attribute=obj_name_attribute,
        )
        if existing_object:
            self.tag_objects_v2(
                session,
                [existing_object],
                project,
                tag,
                obj_name_attribute=obj_name_attribute,
            )
            return existing_object

        return None

    def _validate_and_enrich_record_for_creation(
        self,
        session,
        new_object,
        db_class,
        project,
        versioned,
    ):
        object_type = new_object.__class__.__name__

        object_dict = new_object.dict(exclude_none=True)
        hash_key = fill_object_hash(object_dict, "uid", new_object.metadata.tag)

        if versioned:
            uid = hash_key
        else:
            uid = f"{unversioned_tagged_object_uid_prefix}{new_object.metadata.tag}"
            object_dict["metadata"]["uid"] = uid

        existing_object = self._get_class_instance_by_uid(
            session, db_class, new_object.metadata.name, project, uid
        )
        if existing_object:
            object_uri = generate_object_uri(
                project, new_object.metadata.name, new_object.metadata.tag
            )
            raise mlrun.errors.MLRunConflictError(
                f"Adding an already-existing {object_type} - {object_uri}"
            )

        return uid, new_object.metadata.tag, object_dict

    def _delete_feature_sets(self, session: Session, project: str):
        logger.debug("Removing feature-sets from db", project=project)
        for feature_set_name in self._list_project_feature_set_names(session, project):
            self.delete_feature_set(session, project, feature_set_name)

    def _list_project_feature_set_names(
        self, session: Session, project: str
    ) -> list[str]:
        return [
            name
            for (name,) in self._query(
                session, distinct(FeatureSet.name), project=project
            ).all()
        ]

    def delete_feature_set(self, session, project, name, tag=None, uid=None):
        self._delete_tagged_object(
            session,
            FeatureSet,
            project=project,
            tag=tag,
            uid=uid,
            name=name,
        )

    # ---- Feature Vectors ----
    def create_feature_vector(
        self,
        session,
        project,
        feature_vector: mlrun.common.schemas.FeatureVector,
        versioned=True,
    ) -> str:
        (
            uid,
            tag,
            feature_vector_dict,
        ) = self._validate_and_enrich_record_for_creation(
            session, feature_vector, FeatureVector, project, versioned
        )

        db_feature_vector = FeatureVector(project=project)

        self._update_db_record_from_object_dict(
            db_feature_vector, feature_vector_dict, uid
        )

        self._upsert(session, [db_feature_vector])
        self.tag_objects_v2(session, [db_feature_vector], project, tag)

        return uid

    def get_feature_vector(
        self, session, project: str, name: str, tag: str = None, uid: str = None
    ) -> mlrun.common.schemas.FeatureVector:
        feature_vector = self._get_feature_vector(session, project, name, tag, uid)
        if not feature_vector:
            feature_vector_uri = generate_object_uri(project, name, tag)
            raise mlrun.errors.MLRunNotFoundError(
                f"Feature-vector not found {feature_vector_uri}"
            )

        return feature_vector

    def _get_feature_vector(
        self,
        session,
        project: str,
        name: str,
        tag: str = None,
        uid: str = None,
    ):
        (
            computed_tag,
            feature_vector_tag_uid,
            db_feature_vector,
        ) = self._get_record_by_name_tag_and_uid(
            session, FeatureVector, project, name, tag, uid
        )
        if db_feature_vector:
            feature_vector = self._transform_feature_vector_model_to_schema(
                db_feature_vector
            )

            # If connected to a tag add it to metadata
            if feature_vector_tag_uid:
                feature_vector.metadata.tag = computed_tag
            return feature_vector
        else:
            return None

    def list_feature_vectors(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        state: str = None,
        labels: list[str] = None,
        partition_by: mlrun.common.schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: mlrun.common.schemas.SortField = None,
        partition_order: mlrun.common.schemas.OrderType = mlrun.common.schemas.OrderType.desc,
    ) -> mlrun.common.schemas.FeatureVectorsOutput:
        obj_id_tags = self._get_records_to_tags_map(
            session, FeatureVector, project, tag, name
        )

        # Query the actual objects to be returned
        query = self._query(session, FeatureVector, project=project, state=state)

        if name is not None:
            query = query.filter(
                generate_query_predicate_for_name(FeatureVector.name, name)
            )
        if tag:
            query = query.filter(FeatureVector.id.in_(obj_id_tags.keys()))
        if labels:
            query = self._add_labels_filter(session, query, FeatureVector, labels)

        if partition_by:
            self._assert_partition_by_parameters(
                mlrun.common.schemas.FeatureStorePartitionByField,
                partition_by,
                partition_sort_by,
            )
            query = self._create_partitioned_query(
                session,
                query,
                FeatureVector,
                partition_by,
                rows_per_partition,
                partition_sort_by,
                partition_order,
            )

        feature_vectors = []
        for feature_vector_record in query:
            feature_vectors.extend(
                self._generate_records_with_tags_assigned(
                    feature_vector_record,
                    self._transform_feature_vector_model_to_schema,
                    obj_id_tags,
                    tag,
                )
            )
        return mlrun.common.schemas.FeatureVectorsOutput(
            feature_vectors=feature_vectors
        )

    def list_feature_vectors_tags(
        self,
        session,
        project: str,
    ):
        query = (
            session.query(FeatureVector.name, FeatureVector.Tag.name)
            .filter(FeatureVector.Tag.project == project)
            .join(FeatureVector, FeatureVector.Tag.obj_id == FeatureVector.id)
            .distinct()
        )
        return [(project, row[0], row[1]) for row in query]

    @retry_on_conflict
    def store_feature_vector(
        self,
        session,
        project,
        name,
        feature_vector: mlrun.common.schemas.FeatureVector,
        tag=None,
        uid=None,
        versioned=True,
        always_overwrite=False,
    ) -> str:
        return self._store_tagged_object(
            session,
            FeatureVector,
            project,
            name,
            feature_vector,
            tag=tag,
            uid=uid,
            versioned=versioned,
            always_overwrite=always_overwrite,
        )

    def patch_feature_vector(
        self,
        session,
        project,
        name,
        feature_vector_update: dict,
        tag=None,
        uid=None,
        patch_mode: mlrun.common.schemas.PatchMode = mlrun.common.schemas.PatchMode.replace,
    ) -> str:
        feature_vector_record = self._get_feature_vector(
            session, project, name, tag, uid
        )
        if not feature_vector_record:
            feature_vector_uri = generate_object_uri(project, name, tag)
            raise mlrun.errors.MLRunNotFoundError(
                f"Feature-vector not found {feature_vector_uri}"
            )

        feature_vector_struct = feature_vector_record.dict(exclude_none=True)
        # using mergedeep for merging the patch content into the existing dictionary
        strategy = patch_mode.to_mergedeep_strategy()
        mergedeep.merge(feature_vector_struct, feature_vector_update, strategy=strategy)

        versioned = feature_vector_record.metadata.uid is not None

        feature_vector = mlrun.common.schemas.FeatureVector(**feature_vector_struct)
        return self.store_feature_vector(
            session,
            project,
            name,
            feature_vector,
            feature_vector.metadata.tag,
            uid,
            versioned,
            always_overwrite=True,
        )

    def delete_feature_vector(self, session, project, name, tag=None, uid=None):
        self._delete_tagged_object(
            session,
            FeatureVector,
            project=project,
            tag=tag,
            uid=uid,
            name=name,
        )

    def _delete_tagged_object(
        self,
        session,
        cls,
        project,
        tag=None,
        uid=None,
        name=None,
        key=None,
        commit=True,
        **kwargs,
    ):
        if tag and uid:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Both uid and tag specified when deleting an object."
            )

        # "key" is only used for artifact objects, and "name" is used for all other tagged objects.
        # thus only one should be passed
        if name and key:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Both name and key specified when deleting an object."
            )
        if not name and not key:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Neither name nor key specified when deleting an object."
            )

        obj_name = name or key
        object_id = None

        if uid or tag:
            # try to find the object by given arguments
            query = self._query(
                session,
                cls,
                project=project,
                uid=uid,
                name=name,
                key=key,
                **kwargs,
            )

            # join on tags if given
            if tag and tag != "*":
                query = query.join(cls.Tag, cls.Tag.obj_id == cls.id)
                query = query.filter(cls.Tag.name == tag)

            object_record = query.one_or_none()

            if object_record is None:
                # object not found, nothing to delete
                return None, None

            # get the object id from the object record
            object_id = object_record.id

        if object_id:
            if not commit:
                return "id", object_id
            # deleting tags, because in sqlite the relationships aren't necessarily cascading
            self._delete(session, cls.Tag, obj_id=object_id)
            self._delete(session, cls, id=object_id)
        else:
            if not commit:
                if name:
                    return "name", obj_name
                return "key", obj_name
            # If we got here, neither tag nor uid were provided - delete all references by name.
            # deleting tags, because in sqlite the relationships aren't necessarily cascading
            identifier = {"name": obj_name} if name else {"key": obj_name}
            self._delete(session, cls.Tag, project=project, obj_name=obj_name)
            self._delete(session, cls, project=project, **identifier)

    def _resolve_class_tag_uid(self, session, cls, project, obj_name, tag_name):
        for tag in self._query(
            session, cls.Tag, project=project, obj_name=obj_name, name=tag_name
        ):
            return self._query(session, cls).get(tag.obj_id).uid
        return None

    def _resolve_class_tag_uids(
        self, session, cls, project, tag_name, obj_name=None
    ) -> list[str]:
        uids = []

        query = self._query(session, cls.Tag, project=project, name=tag_name)
        if obj_name:
            query = query.filter(
                generate_query_predicate_for_name(cls.Tag.obj_name, obj_name)
            )

        for tag in query:
            # TODO: query db in a single call
            obj = self._query(session, cls).get(tag.obj_id)
            if obj:
                uids.append(obj.uid)
        return uids

    def _attach_most_recent_artifact_query(self, session, query):
        # Create a sub query of latest uid (by updated) per (project,key)
        subq = (
            session.query(
                ArtifactV2.project,
                ArtifactV2.key,
                func.max(ArtifactV2.updated).label("max_updated"),
            )
            .group_by(
                ArtifactV2.project,
                ArtifactV2.key,
            )
            .subquery()
        )

        # Join current query with sub query on (project, key)
        return query.join(
            subq,
            and_(
                ArtifactV2.project == subq.c.project,
                ArtifactV2.key == subq.c.key,
                ArtifactV2.updated == subq.c.max_updated,
            ),
        )

    def _query(self, session, cls, **kw):
        kw = {k: v for k, v in kw.items() if v is not None}
        return session.query(cls).filter_by(**kw)

    def _get_count(self, session, cls):
        return session.query(func.count(inspect(cls).primary_key[0])).scalar()

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
                raise mlrun.errors.MLRunConflictError(
                    f"Failed to add user: {err_to_str(err)}"
                ) from err
        return users

    def _get_class_instance_by_uid(self, session, cls, name, project, uid):
        query = self._query(session, cls, name=name, project=project, uid=uid)
        return query.one_or_none()

    def _get_run(self, session, uid, project, iteration, with_for_update=False):
        query = self._query(session, Run, uid=uid, project=project, iteration=iteration)
        if with_for_update:
            query = query.populate_existing().with_for_update()

        return query.one_or_none()

    def _delete_empty_labels(self, session, cls):
        session.query(cls).filter(cls.parent == NULL).delete()
        session.commit()

    def _upsert(self, session, objects, ignore=False):
        if not objects:
            return
        for object_ in objects:
            session.add(object_)
        self._commit(session, objects, ignore)

    @staticmethod
    def _commit(session, objects, ignore=False):
        def _try_commit_obj():
            try:
                session.commit()
            except SQLAlchemyError as sql_err:
                session.rollback()
                classes = list(set([object_.__class__.__name__ for object_ in objects]))

                # if the database is locked, we raise a retryable error
                if "database is locked" in str(sql_err):
                    logger.warning(
                        "Database is locked. Retrying",
                        classes_to_commit=classes,
                        err=str(sql_err),
                    )
                    raise mlrun.errors.MLRunRuntimeError(
                        "Failed committing changes, database is locked"
                    ) from sql_err

                # the error is not retryable, so we try to identify weather there was a conflict or not
                # either way - we wrap the error with a fatal error so the retry mechanism will stop
                logger.warning(
                    "Failed committing changes to DB",
                    classes=classes,
                    err=err_to_str(sql_err),
                )
                if not ignore:
                    # get the identifiers of the objects that failed to commit, for logging purposes
                    identifiers = ",".join(
                        object_.get_identifier_string() for object_ in objects
                    )

                    mlrun_error = mlrun.errors.MLRunRuntimeError(
                        f"Failed committing changes to DB. classes={classes} objects={identifiers}"
                    )

                    # check if the error is a conflict error
                    if any([message in str(sql_err) for message in conflict_messages]):
                        mlrun_error = mlrun.errors.MLRunConflictError(
                            f"Conflict - at least one of the objects already exists: {identifiers}"
                        )

                    # we want to keep the exception stack trace, but we also want the retry mechanism to stop
                    # so, we raise a new indicative exception from the original sql exception (this keeps
                    # the stack trace intact), and then wrap it with a fatal error (which stops the retry mechanism).
                    # Note - this way, the exception is raised from this code section, and not from the retry function.
                    try:
                        raise mlrun_error from sql_err
                    except (
                        mlrun.errors.MLRunRuntimeError,
                        mlrun.errors.MLRunConflictError,
                    ) as exc:
                        raise mlrun.errors.MLRunFatalFailureError(
                            original_exception=exc
                        )

        if config.httpdb.db.commit_retry_timeout:
            mlrun.utils.helpers.retry_until_successful(
                config.httpdb.db.commit_retry_interval,
                config.httpdb.db.commit_retry_timeout,
                logger,
                False,
                _try_commit_obj,
            )

    def _find_runs(self, session, uid, project, labels):
        labels = label_set(labels)
        if project == "*":
            project = None
        query = self._query(session, Run, project=project)
        if uid:
            # uid may be either a single uid (string) or a list of uids
            uid = mlrun.utils.helpers.as_list(uid)
            query = query.filter(Run.uid.in_(uid))
        return self._add_labels_filter(session, query, Run, labels)

    def _get_db_notifications(
        self, session, cls, name: str = None, parent_id: str = None, project: str = None
    ):
        return self._query(
            session, cls.Notification, name=name, parent_id=parent_id, project=project
        ).all()

    @staticmethod
    def _escape_characters_for_like_query(value: str) -> str:
        return (
            value.translate(value.maketrans({"_": r"\_", "%": r"\%"})) if value else ""
        )

    def _find_functions(
        self,
        session: Session,
        name: str,
        project: str,
        labels: typing.Union[str, list[str], None] = None,
        tag: typing.Optional[str] = None,
        hash_key: typing.Optional[str] = None,
        since: datetime = None,
        until: datetime = None,
        page: typing.Optional[int] = None,
        page_size: typing.Optional[int] = None,
    ) -> list[tuple[Function, str]]:
        """
        Query functions from the DB by the given filters.

        :param session: The DB session.
        :param name: The name of the function to query.
        :param project: The project of the function to query.
        :param labels: The labels of the function to query.
        :param tag: The tag of the function to query.
        :param hash_key: The hash key of the function to query.
        :param since: Filter functions that were updated after this time
        :param until: Filter functions that were updated before this time
        :param page: The page number to query.
        :param page_size: The page size to query.
        """
        query = session.query(Function, Function.Tag.name)
        query = query.filter(Function.project == project)

        if name:
            query = query.filter(generate_query_predicate_for_name(Function.name, name))

        if hash_key is not None:
            query = query.filter(Function.uid == hash_key)

        if since or until:
            since = since or datetime.min
            until = until or datetime.max
            query = query.filter(
                and_(Function.updated >= since, Function.updated <= until)
            )

        if not tag:
            # If no tag is given, we need to outer join to get all functions, even if they don't have tags.
            query = query.outerjoin(Function.Tag, Function.id == Function.Tag.obj_id)
        else:
            # Only get functions that have tags with join (faster than outer join)
            query = query.join(Function.Tag, Function.id == Function.Tag.obj_id)
            if tag != "*":
                # Filter on the specific tag
                query = query.filter(Function.Tag.name == tag)

        labels = label_set(labels)
        query = self._add_labels_filter(session, query, Function, labels)
        query = self._paginate_query(query, page, page_size)
        return query

    def _delete(self, session, cls, **kw):
        query = session.query(cls).filter_by(**kw)
        for obj in query:
            session.delete(obj)
        session.commit()

    def _find_labels(self, session, cls, label_cls, labels):
        return session.query(cls).join(label_cls).filter(label_cls.name.in_(labels))

    def _add_labels_filter(self, session, query, cls, labels):
        if not labels:
            return query

        preds = []
        # Some specific handling is needed for the case of a query like "label=x&label=x=value". In this case
        # of course it should be reduced to "label=x=value". That's why we need to keep the labels that are queried
        # with values, and then remove it from the list of labels queried without value.
        label_names_with_values = set()
        label_names_no_values = set()

        for lbl in labels:
            if "=" in lbl:
                name, value = (v.strip() for v in lbl.split("=", 1))
                cond = and_(
                    generate_query_predicate_for_name(cls.Label.name, name),
                    generate_query_predicate_for_name(cls.Label.value, value),
                )
                preds.append(cond)
                label_names_with_values.add(name)
            else:
                label_names_no_values.add(lbl.strip())

        for name in label_names_no_values.difference(label_names_with_values):
            preds.append(generate_query_predicate_for_name(cls.Label.name, name))

        if len(preds) == 1:
            # A single label predicate is a common case, and there's no need to burden the DB with
            # a more complex query for that case.
            subq = session.query(cls.Label).filter(*preds).subquery("labels")
        else:
            # Basically do an "or" query on the predicates, and count how many rows each parent object has -
            # if it has as much rows as predicates, then it means it answers all the conditions.
            subq = (
                session.query(cls.Label)
                .filter(or_(*preds))
                .group_by(cls.Label.parent)
                .having(func.count(cls.Label.parent) == len(preds))
                .subquery("labels")
            )

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

    def _transform_schedule_record_to_scheme(
        self,
        schedule_record: Schedule,
    ) -> mlrun.common.schemas.ScheduleRecord:
        schedule = mlrun.common.schemas.ScheduleRecord.from_orm(schedule_record)
        schedule.creation_time = self._add_utc_timezone(schedule.creation_time)
        schedule.next_run_time = self._add_utc_timezone(schedule.next_run_time)
        return schedule

    @staticmethod
    def _add_utc_timezone(time_value: typing.Optional[datetime]):
        """
        sqlalchemy losing timezone information with sqlite so we're returning it
        https://stackoverflow.com/questions/6991457/sqlalchemy-losing-timezone-information-with-sqlite
        """
        if time_value:
            if time_value.tzinfo is None:
                return pytz.utc.localize(time_value)
        return time_value

    @staticmethod
    def _transform_feature_set_model_to_schema(
        feature_set_record: FeatureSet,
        tag=None,
        format_: mlrun.common.formatters.FeatureSetFormat = mlrun.common.formatters.FeatureSetFormat.full,
    ) -> mlrun.common.schemas.FeatureSet:
        feature_set_full_dict = feature_set_record.full_object
        feature_set_full_dict = mlrun.common.formatters.FeatureSetFormat.format_obj(
            feature_set_full_dict, format_
        )
        feature_set_resp = mlrun.common.schemas.FeatureSet(**feature_set_full_dict)

        feature_set_resp.metadata.tag = tag
        return feature_set_resp

    @staticmethod
    def _transform_feature_vector_model_to_schema(
        feature_vector_record: FeatureVector, tag=None, format_=None
    ) -> mlrun.common.schemas.FeatureVector:
        feature_vector_full_dict = feature_vector_record.full_object
        feature_vector_resp = mlrun.common.schemas.FeatureVector(
            **feature_vector_full_dict
        )

        feature_vector_resp.metadata.tag = tag
        feature_vector_resp.metadata.created = feature_vector_record.created
        return feature_vector_resp

    def _transform_project_record_to_schema(
        self, session: Session, project_record: Project
    ) -> mlrun.common.schemas.ProjectOut:
        # in projects that was created before 0.6.0 the full object wasn't created properly - fix that, and return
        if not project_record.full_object:
            project = mlrun.common.schemas.Project(
                metadata=mlrun.common.schemas.ProjectMetadata(
                    name=project_record.name,
                    created=project_record.created,
                ),
                spec=mlrun.common.schemas.ProjectSpec(
                    description=project_record.description,
                    source=project_record.source,
                    default_function_node_selector=project_record.default_function_node_selector,
                ),
                status=mlrun.common.schemas.ObjectStatus(
                    state=project_record.state,
                ),
            )
            self.store_project(session, project_record.name, project)
            return mlrun.common.schemas.ProjectOut(**project.dict())
        # TODO: handle transforming the functions/workflows/artifacts references to real objects
        return mlrun.common.schemas.ProjectOut(**project_record.full_object)

    def _transform_notification_record_to_spec_and_status(
        self,
        notification_record,
    ) -> tuple[dict, dict]:
        notification_spec = self._transform_notification_record_to_schema(
            notification_record
        ).to_dict()
        notification_status = {
            "status": notification_spec.pop("status", None),
            "sent_time": notification_spec.pop("sent_time", None),
            "reason": notification_spec.pop("reason", None),
        }
        return notification_spec, notification_status

    @staticmethod
    def _transform_notification_record_to_schema(
        notification_record,
    ) -> mlrun.model.Notification:
        return mlrun.model.Notification(
            kind=notification_record.kind,
            name=notification_record.name,
            message=notification_record.message,
            severity=notification_record.severity,
            when=notification_record.when.split(","),
            condition=notification_record.condition,
            secret_params=notification_record.secret_params,
            params=notification_record.params,
            status=notification_record.status,
            sent_time=notification_record.sent_time,
            reason=notification_record.reason,
        )

    def _move_and_reorder_table_items(
        self, session, moved_object, move_to=None, move_from=None
    ):
        # If move_to is None - delete object. If move_from is None - insert a new object
        moved_object.index = move_to

        if move_from == move_to:
            # It's just modifying the same object - update and exit.
            # using merge since primary key is changing
            session.merge(moved_object)
            session.commit()
            return

        modifier = 1
        if move_from is None:
            start, end = move_to, None
        elif move_to is None:
            start, end = move_from + 1, None
            modifier = -1
        else:
            if move_from < move_to:
                start, end = move_from + 1, move_to
                modifier = -1
            else:
                start, end = move_to, move_from - 1

        query = session.query(HubSource).filter(HubSource.index >= start)
        if end:
            query = query.filter(HubSource.index <= end)

        for source_record in query:
            source_record.index = source_record.index + modifier
            # using merge since primary key is changing
            session.merge(source_record)

        if move_to:
            # using merge since primary key is changing
            session.merge(moved_object)
        else:
            session.delete(moved_object)
        session.commit()

    @staticmethod
    def _transform_hub_source_record_to_schema(
        hub_source_record: HubSource,
    ) -> mlrun.common.schemas.IndexedHubSource:
        source_full_dict = hub_source_record.full_object
        hub_source = mlrun.common.schemas.HubSource(**source_full_dict)
        return mlrun.common.schemas.IndexedHubSource(
            index=hub_source_record.index, source=hub_source
        )

    @staticmethod
    def _transform_hub_source_schema_to_record(
        hub_source_schema: mlrun.common.schemas.IndexedHubSource,
        current_object: HubSource = None,
    ):
        now = datetime.now(timezone.utc)
        if current_object:
            if current_object.name != hub_source_schema.source.metadata.name:
                raise mlrun.errors.MLRunInternalServerError(
                    "Attempt to update object while replacing its name"
                )
            created_timestamp = current_object.created
        else:
            created_timestamp = hub_source_schema.source.metadata.created or now
        updated_timestamp = hub_source_schema.source.metadata.updated or now

        hub_source_record = HubSource(
            id=current_object.id if current_object else None,
            name=hub_source_schema.source.metadata.name,
            index=hub_source_schema.index,
            created=created_timestamp,
            updated=updated_timestamp,
        )
        full_object = hub_source_schema.source.dict()
        full_object["metadata"]["created"] = str(created_timestamp)
        full_object["metadata"]["updated"] = str(updated_timestamp)
        # Make sure we don't keep any credentials in the DB. These are handled in the hub crud object.
        full_object["spec"].pop("credentials", None)

        hub_source_record.full_object = full_object
        return hub_source_record

    @staticmethod
    def _validate_and_adjust_hub_order(session, order):
        max_order = session.query(func.max(HubSource.index)).scalar()
        if not max_order or max_order < 0:
            max_order = 0

        if order == mlrun.common.schemas.hub.last_source_index:
            order = max_order + 1

        if order > max_order + 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Order must not exceed the current maximal order + 1. max_order = {max_order}, order = {order}"
            )
        if order < 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Order of inserted source must be greater than 0 or "
                + f"{mlrun.common.schemas.hub.last_source_index} (for last). order = {order}"
            )
        return order

    # ---- Hub Sources ----
    def create_hub_source(
        self, session, ordered_source: mlrun.common.schemas.IndexedHubSource
    ):
        logger.debug(
            "Creating hub source in DB",
            index=ordered_source.index,
            name=ordered_source.source.metadata.name,
        )

        order = self._validate_and_adjust_hub_order(session, ordered_source.index)
        name = ordered_source.source.metadata.name
        source_record = self._query(session, HubSource, name=name).one_or_none()
        if source_record:
            raise mlrun.errors.MLRunConflictError(
                f"Hub source name already exists. name = {name}"
            )
        source_record = self._transform_hub_source_schema_to_record(ordered_source)

        self._move_and_reorder_table_items(
            session, source_record, move_to=order, move_from=None
        )

    @retry_on_conflict
    def store_hub_source(
        self,
        session,
        name,
        ordered_source: mlrun.common.schemas.IndexedHubSource,
    ):
        logger.debug("Storing hub source in DB", index=ordered_source.index, name=name)

        if name != ordered_source.source.metadata.name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Conflict between resource name and metadata.name in the stored object"
            )
        order = self._validate_and_adjust_hub_order(session, ordered_source.index)

        source_record = self._query(session, HubSource, name=name).one_or_none()
        current_order = source_record.index if source_record else None
        if current_order == mlrun.common.schemas.hub.last_source_index:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Attempting to modify the global hub source."
            )
        source_record = self._transform_hub_source_schema_to_record(
            ordered_source, source_record
        )

        self._move_and_reorder_table_items(
            session, source_record, move_to=order, move_from=current_order
        )

    def list_hub_sources(self, session) -> list[mlrun.common.schemas.IndexedHubSource]:
        results = []
        query = self._query(session, HubSource).order_by(HubSource.index.desc())
        for record in query:
            ordered_source = self._transform_hub_source_record_to_schema(record)
            # Need this to make the list return such that the default source is last in the response.
            if ordered_source.index != mlrun.common.schemas.last_source_index:
                results.insert(0, ordered_source)
            else:
                results.append(ordered_source)
        return results

    def _list_hub_sources_without_transform(self, session) -> list[HubSource]:
        return self._query(session, HubSource).all()

    def delete_hub_source(self, session, name):
        logger.debug("Deleting hub source from DB", name=name)

        source_record = self._query(session, HubSource, name=name).one_or_none()
        if not source_record:
            return

        current_order = source_record.index
        if current_order == mlrun.common.schemas.hub.last_source_index:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Attempting to delete the global hub source."
            )

        self._move_and_reorder_table_items(
            session, source_record, move_to=None, move_from=current_order
        )

    def get_hub_source(
        self, session, name=None, index=None, raise_on_not_found=True
    ) -> typing.Optional[mlrun.common.schemas.IndexedHubSource]:
        source_record = self._query(
            session, HubSource, name=name, index=index
        ).one_or_none()
        if not source_record:
            log_method = logger.warning if raise_on_not_found else logger.debug
            message = f"Hub source not found. name = {name}"
            log_method(message)
            if raise_on_not_found:
                raise mlrun.errors.MLRunNotFoundError(message)

            return None

        return self._transform_hub_source_record_to_schema(source_record)

    # ---- Data Versions ----
    def get_current_data_version(
        self, session, raise_on_not_found=True
    ) -> typing.Optional[str]:
        current_data_version_record = (
            self._query(session, DataVersion)
            .order_by(DataVersion.created.desc())
            .limit(1)
            .one_or_none()
        )
        if not current_data_version_record:
            log_method = logger.warning if raise_on_not_found else logger.debug
            message = "No data version found"
            log_method(message)
            if raise_on_not_found:
                raise mlrun.errors.MLRunNotFoundError(message)
        if current_data_version_record:
            return current_data_version_record.version
        else:
            return None

    def create_data_version(self, session, version):
        logger.debug(
            "Creating data version in DB",
            version=version,
        )

        now = datetime.now(timezone.utc)
        data_version_record = DataVersion(version=version, created=now)
        self._upsert(session, [data_version_record])

    def store_alert_template(
        self, session, template: mlrun.common.schemas.AlertTemplate
    ) -> mlrun.common.schemas.AlertTemplate:
        template_record = self._get_alert_template_record(
            session, template.template_name
        )
        if not template_record:
            return self._create_alert_template(session, template)
        template_record.full_object = template.dict()

        self._upsert(session, [template_record])
        return self._transform_alert_template_record_to_schema(
            self._get_alert_template_record(session, template.template_name)
        )

    def _create_alert_template(
        self, session, template: mlrun.common.schemas.AlertTemplate
    ) -> mlrun.common.schemas.AlertTemplate:
        template_record = self._transform_alert_template_schema_to_record(template)
        self._upsert(session, [template_record])
        return self._transform_alert_template_record_to_schema(template_record)

    def delete_alert_template(self, session, name: str):
        self._delete(session, AlertTemplate, name=name)

    def list_alert_templates(self, session) -> list[mlrun.common.schemas.AlertTemplate]:
        query = self._query(session, AlertTemplate)
        return list(map(self._transform_alert_template_record_to_schema, query.all()))

    def get_alert_template(
        self, session, name: str
    ) -> mlrun.common.schemas.AlertTemplate:
        return self._transform_alert_template_record_to_schema(
            self._get_alert_template_record(session, name)
        )

    def get_all_alerts(self, session) -> list[mlrun.common.schemas.AlertConfig]:
        query = self._query(session, AlertConfig)
        return list(map(self._transform_alert_config_record_to_schema, query.all()))

    def get_num_configured_alerts(self, session) -> int:
        return self._get_count(session, AlertConfig)

    def store_alert(
        self, session, alert: mlrun.common.schemas.AlertConfig
    ) -> mlrun.common.schemas.AlertConfig:
        alert_record = self._get_alert_record(session, alert.name, alert.project)
        if not alert_record:
            return self._create_alert(session, alert)
        alert_record.full_object = alert.dict()
        alert_state = self.get_alert_state(session, alert_record.id)

        self._delete_alert_notifications(session, alert.name, alert, alert.project)
        self._store_notifications(
            session,
            AlertConfig,
            alert.get_raw_notifications(),
            alert_record.id,
            alert.project,
        )

        self._upsert(session, [alert_record, alert_state])
        return self.get_alert_by_id(session, alert_record.id)

    def _create_alert(
        self, session, alert: mlrun.common.schemas.AlertConfig
    ) -> mlrun.common.schemas.AlertConfig:
        alert_record = self._transform_alert_config_schema_to_record(alert)
        self._upsert(session, [alert_record])

        alert_record = self._get_alert_record(
            session, alert_record.name, alert_record.project
        )

        self._store_notifications(
            session,
            AlertConfig,
            alert.get_raw_notifications(),
            alert_record.id,
            alert.project,
        )
        self.create_alert_state(session, alert_record)

        return self._transform_alert_config_record_to_schema(alert_record)

    def delete_alert(self, session, project: str, name: str):
        self._delete(session, AlertConfig, project=project, name=name)

    def list_alerts(
        self, session, project: str = None
    ) -> list[mlrun.common.schemas.AlertConfig]:
        query = self._query(session, AlertConfig)

        if project and project != "*":
            query = query.filter(AlertConfig.project == project)

        alerts = list(map(self._transform_alert_config_record_to_schema, query.all()))
        for alert in alerts:
            self.enrich_alert(session, alert)
        return alerts

    def get_alert(
        self, session, project: str, name: str
    ) -> mlrun.common.schemas.AlertConfig:
        return self._transform_alert_config_record_to_schema(
            self._get_alert_record(session, name, project)
        )

    def get_alert_by_id(
        self, session, alert_id: int
    ) -> mlrun.common.schemas.AlertConfig:
        return self._transform_alert_config_record_to_schema(
            self._get_alert_record_by_id(session, alert_id)
        )

    def enrich_alert(self, session, alert: mlrun.common.schemas.AlertConfig):
        state = self.get_alert_state(session, alert.id)
        alert.state = (
            mlrun.common.schemas.AlertActiveState.ACTIVE
            if state.active
            else mlrun.common.schemas.AlertActiveState.INACTIVE
        )
        alert.count = state.count
        alert.created = state.created

        def _enrich_notification(_notification):
            _notification = _notification.to_dict()
            # we don't want to return the secret
            del _notification["secret_params"]

            if not isinstance(_notification["when"], list):
                _notification["when"] = [_notification["when"]]
            return _notification

        notifications = [
            mlrun.common.schemas.notification.Notification(
                **_enrich_notification(notification)
            )
            for notification in self._get_db_notifications(
                session, AlertConfig, parent_id=alert.id
            )
        ]

        cooldowns = [
            notification.cooldown_period for notification in alert.notifications
        ]

        alert.notifications = [
            mlrun.common.schemas.alert.AlertNotification(
                cooldown_period=cooldown, notification=notification
            )
            for cooldown, notification in zip(cooldowns, notifications)
        ]

    @staticmethod
    def _transform_alert_template_schema_to_record(
        alert_template: mlrun.common.schemas.AlertTemplate,
    ) -> AlertTemplate:
        template_record = AlertTemplate(
            id=alert_template.template_id,
            name=alert_template.template_name,
        )
        template_record.full_object = alert_template.dict()
        return template_record

    @staticmethod
    def _transform_alert_template_record_to_schema(
        template_record: AlertTemplate,
    ) -> mlrun.common.schemas.AlertTemplate:
        if template_record is None:
            return None

        template = mlrun.common.schemas.AlertTemplate(**template_record.full_object)
        template.template_id = template_record.id
        return template

    @staticmethod
    def _transform_alert_config_record_to_schema(
        alert_config_record: AlertConfig,
    ) -> mlrun.common.schemas.AlertConfig:
        if alert_config_record is None:
            return None

        alert = mlrun.common.schemas.AlertConfig(**alert_config_record.full_object)
        alert.id = alert_config_record.id
        return alert

    @staticmethod
    def _transform_alert_config_schema_to_record(
        alert: mlrun.common.schemas.AlertConfig,
    ) -> AlertConfig:
        alert_record = AlertConfig(
            id=alert.id,
            name=alert.name,
            project=alert.project,
        )
        alert_record.full_object = alert.dict()
        return alert_record

    def _get_alert_template_record(self, session, name: str) -> AlertTemplate:
        return self._query(session, AlertTemplate, name=name).one_or_none()

    def _get_alert_record(self, session, name: str, project: str) -> AlertConfig:
        return self._query(
            session, AlertConfig, name=name, project=project
        ).one_or_none()

    def _get_alert_record_by_id(self, session, alert_id: int) -> AlertConfig:
        return self._query(session, AlertConfig, id=alert_id).one_or_none()

    def store_alert_state(
        self,
        session,
        project: str,
        name: str,
        last_updated: datetime,
        count: typing.Optional[int] = None,
        active: bool = False,
        obj: typing.Optional[dict] = None,
    ):
        alert = self.get_alert(session, project, name)
        state = self.get_alert_state(session, alert.id)
        if count is not None:
            state.count = count
        state.last_updated = last_updated
        state.active = active
        if obj is not None:
            state.full_object = obj
        self._upsert(session, [state])

    def get_alert_state(self, session, alert_id: int) -> AlertState:
        return self._query(session, AlertState, parent_id=alert_id).one()

    def get_alert_state_dict(self, session, alert_id: int) -> dict:
        state = self.get_alert_state(session, alert_id)
        if state is not None:
            return state.to_dict()

    def create_alert_state(self, session, alert_record):
        state = AlertState(count=0, parent_id=alert_record.id)
        self._upsert(session, [state])

    def delete_alert_notifications(
        self,
        session,
        project: str,
    ):
        if resp := self._query(session, AlertConfig, project=project).all():
            for alert in resp:
                self._delete_alert_notifications(
                    session, alert.name, alert, project, commit=False
                )
                session.delete(alert)

            session.commit()

    def _delete_alert_notifications(
        self, session, name: str, alert: AlertConfig, project: str, commit: bool = True
    ):
        query = self._get_db_notifications(
            session, AlertConfig, None, alert.id, project
        )
        for notification in query:
            session.delete(notification)

        if commit:
            session.commit()

    # ---- Background Tasks ----

    @retry_on_conflict
    def store_background_task(
        self,
        session,
        name: str,
        project: str,
        state: str = mlrun.common.schemas.BackgroundTaskState.running,
        timeout: int = None,
        error: str = None,
    ):
        error = server.api.db.sqldb.helpers.ensure_max_length(error)
        background_task_record = self._query(
            session,
            BackgroundTask,
            name=name,
            project=project,
        ).one_or_none()
        now = mlrun.utils.now_date()
        if background_task_record:
            # we don't want to be able to change state after it reached terminal state
            if (
                background_task_record.state
                in mlrun.common.schemas.BackgroundTaskState.terminal_states()
                and state != background_task_record.state
            ):
                raise mlrun.errors.MLRunRuntimeError(
                    "Background task already reached terminal state, can not change to another state. Failing"
                )

            if timeout and mlrun.mlconf.background_tasks.timeout_mode == "enabled":
                background_task_record.timeout = int(timeout)
            background_task_record.state = state
            background_task_record.error = error
            background_task_record.updated = now
        else:
            if mlrun.mlconf.background_tasks.timeout_mode == "disabled":
                timeout = None

            background_task_record = BackgroundTask(
                name=name,
                project=project,
                state=state,
                created=now,
                updated=now,
                timeout=int(timeout) if timeout else None,
                error=error,
            )
        self._upsert(session, [background_task_record])

    def get_background_task(
        self,
        session: Session,
        name: str,
        project: str,
        background_task_exceeded_timeout_func,
    ) -> mlrun.common.schemas.BackgroundTask:
        background_task_record = self._get_background_task_record(
            session, name, project
        )
        background_task_record = self._apply_background_task_timeout(
            session,
            background_task_exceeded_timeout_func,
            background_task_record,
        )

        return self._transform_background_task_record_to_schema(background_task_record)

    def list_background_tasks(
        self,
        session,
        project: str,
        background_task_exceeded_timeout_func,
        states: typing.Optional[list[str]] = None,
        created_from: datetime = None,
        created_to: datetime = None,
        last_update_time_from: datetime = None,
        last_update_time_to: datetime = None,
    ) -> list[mlrun.common.schemas.BackgroundTask]:
        background_tasks = []
        query = self._list_project_background_tasks(session, project)
        if states is not None:
            query = query.filter(BackgroundTask.state.in_(states))
        if created_from is not None:
            query = query.filter(BackgroundTask.created >= created_from)
        if created_to is not None:
            query = query.filter(BackgroundTask.created <= created_to)
        if last_update_time_from is not None:
            query = query.filter(BackgroundTask.updated >= last_update_time_from)
        if last_update_time_to is not None:
            query = query.filter(BackgroundTask.updated <= last_update_time_to)

        background_task_records = query.all()
        for background_task_record in background_task_records:
            background_task_record = self._apply_background_task_timeout(
                session,
                background_task_exceeded_timeout_func,
                background_task_record,
            )

            # retest state after applying timeout
            if states and background_task_record.state not in states:
                continue

            background_tasks.append(
                self._transform_background_task_record_to_schema(background_task_record)
            )

        return background_tasks

    def delete_background_task(self, session: Session, name: str, project: str):
        self._delete(session, BackgroundTask, name=name, project=project)

    def _apply_background_task_timeout(
        self,
        session: Session,
        background_task_exceeded_timeout_func: typing.Callable,
        background_task_record: BackgroundTask,
    ):
        if (
            background_task_exceeded_timeout_func
            and background_task_exceeded_timeout_func(
                background_task_record.updated,
                background_task_record.timeout,
                background_task_record.state,
            )
        ):
            # lazy update of state, only if get background task was requested and the timeout for the update passed
            # and the task still in progress then we change to failed
            self.store_background_task(
                session,
                background_task_record.name,
                background_task_record.project,
                mlrun.common.schemas.background_task.BackgroundTaskState.failed,
            )
            background_task_record = self._get_background_task_record(
                session, background_task_record.name, background_task_record.project
            )
        return background_task_record

    @staticmethod
    def _transform_background_task_record_to_schema(
        background_task_record: BackgroundTask,
    ) -> mlrun.common.schemas.BackgroundTask:
        return mlrun.common.schemas.BackgroundTask(
            metadata=mlrun.common.schemas.BackgroundTaskMetadata(
                name=background_task_record.name,
                project=background_task_record.project,
                created=background_task_record.created,
                updated=background_task_record.updated,
                timeout=background_task_record.timeout,
            ),
            spec=mlrun.common.schemas.BackgroundTaskSpec(),
            status=mlrun.common.schemas.BackgroundTaskStatus(
                state=background_task_record.state,
                error=background_task_record.error,
            ),
        )

    def _list_project_background_task_names(
        self, session: Session, project: str
    ) -> list[str]:
        return [
            name
            for (name,) in self._query(
                session, distinct(BackgroundTask.name), project=project
            ).all()
        ]

    def _list_project_background_tasks(self, session: Session, project: str):
        return self._query(session, BackgroundTask, project=project)

    def _delete_background_tasks(self, session: Session, project: str):
        logger.debug("Removing project background tasks from db", project=project)
        for background_task_name in self._list_project_background_task_names(
            session, project
        ):
            self.delete_background_task(session, background_task_name, project)

    def _get_background_task_record(
        self,
        session: Session,
        name: str,
        project: str,
        raise_on_not_found: bool = True,
    ) -> typing.Optional[BackgroundTask]:
        background_task_record = self._query(
            session, BackgroundTask, name=name, project=project
        ).one_or_none()
        if not background_task_record:
            if not raise_on_not_found:
                return None
            raise mlrun.errors.MLRunNotFoundError(
                f"Background task not found: name={name}, project={project}"
            )
        return background_task_record

    # ---- Run Notifications ----
    def store_run_notifications(
        self,
        session,
        notification_objects: list[mlrun.model.Notification],
        run_uid: str,
        project: str,
    ):
        # iteration is 0, as we don't support multiple notifications per hyper param run, only for the whole run
        run = self._get_run(session, run_uid, project, 0)
        if not run:
            raise mlrun.errors.MLRunNotFoundError(
                f"Run not found: uid={run_uid}, project={project}"
            )

        self._store_notifications(session, Run, notification_objects, run.id, project)

    def store_alert_notifications(
        self,
        session,
        notification_objects: list[mlrun.model.Notification],
        alert_id: str,
        project: str,
    ):
        if self._get_alert_record_by_id(session, alert_id):
            self._store_notifications(
                session, AlertConfig, notification_objects, alert_id, project
            )
        else:
            raise mlrun.errors.MLRunNotFoundError(
                f"Alert not found: uid={alert_id}, project={project}"
            )

    def _store_notifications(
        self,
        session,
        cls,
        notification_objects: list[mlrun.model.Notification],
        parent_id: str,
        project: str,
    ):
        db_notifications = {
            notification.name: notification
            for notification in self._get_db_notifications(
                session, cls, parent_id=parent_id
            )
        }
        notifications = []
        logger.debug(
            "Storing notifications",
            notifications_length=len(notification_objects),
            parent_id=parent_id,
            project=project,
        )
        for notification_model in notification_objects:
            new_notification = False
            notification = db_notifications.get(notification_model.name, None)
            if not notification:
                new_notification = True
                notification = cls.Notification(
                    name=notification_model.name, parent_id=parent_id, project=project
                )

            notification.kind = notification_model.kind
            notification.message = notification_model.message or ""
            notification.severity = (
                notification_model.severity
                or mlrun.common.schemas.NotificationSeverity.INFO
            )
            notification.when = ",".join(notification_model.when or [])
            notification.condition = notification_model.condition or ""
            notification.secret_params = notification_model.secret_params
            notification.params = notification_model.params
            notification.status = (
                notification_model.status
                or mlrun.common.schemas.NotificationStatus.PENDING
            )
            notification.sent_time = notification_model.sent_time
            notification.reason = notification_model.reason

            logger.debug(
                f"Storing {'new' if new_notification else 'existing'} notification",
                notification_name=notification.name,
                notification_status=notification.status,
                parent_id=parent_id,
                project=project,
            )
            notifications.append(notification)

        self._upsert(session, notifications)

    def list_run_notifications(
        self,
        session,
        run_uid: str,
        project: str = "",
    ) -> list[mlrun.model.Notification]:
        # iteration is 0, as we don't support multiple notifications per hyper param run, only for the whole run
        run = self._get_run(session, run_uid, project, 0)
        if not run:
            return []

        return [
            self._transform_notification_record_to_schema(notification)
            for notification in self._query(
                session, Run.Notification, parent_id=run.id
            ).all()
        ]

    def delete_run_notifications(
        self,
        session,
        name: str = None,
        run_uid: str = None,
        project: str = None,
        commit: bool = True,
    ):
        run_id = None
        if run_uid:
            # iteration is 0, as we don't support multiple notifications per hyper param run, only for the whole run
            run = self._get_run(session, run_uid, project, 0)
            if not run:
                raise mlrun.errors.MLRunNotFoundError(
                    f"Run not found: uid={run_uid}, project={project}"
                )
            run_id = run.id

        project = project or config.default_project
        if project == "*":
            project = None

        query = self._get_db_notifications(session, Run, name, run_id, project)
        for notification in query:
            session.delete(notification)

        if commit:
            session.commit()

    def set_run_notifications(
        self,
        session: Session,
        project: str,
        notifications: list[mlrun.model.Notification],
        identifier: mlrun.common.schemas.RunIdentifier,
        **kwargs,
    ):
        """
        Set notifications for a run. This will replace any existing notifications.
        :param session: SQLAlchemy session
        :param project: Project name
        :param notifications: List of notifications to set
        :param identifier: Run identifier
        :param kwargs: Ignored additional arguments (for interfacing purposes)
        """
        run = self._get_run(session, identifier.uid, project, None)
        if not run:
            raise mlrun.errors.MLRunNotFoundError(
                f"Run not found: project={project}, uid={identifier.uid}"
            )

        run.struct.setdefault("spec", {})["notifications"] = [
            notification.to_dict() for notification in notifications
        ]

        # update run, delete and store notifications all in one transaction.
        # using session.add instead of upsert, so we don't commit the run.
        # the commit will happen at the end (in store_run_notifications, or manually at the end).
        session.add(run)
        self.delete_run_notifications(
            session, run_uid=run.uid, project=project, commit=False
        )
        if notifications:
            self.store_run_notifications(
                session,
                notification_objects=notifications,
                run_uid=run.uid,
                project=project,
            )
        self._commit(session, [run], ignore=True)

    # ---- Data Store ----
    def store_datastore_profile(
        self, session, info: mlrun.common.schemas.DatastoreProfile
    ):
        """
        Create or replace a datastore profile.
        :param session: SQLAlchemy session
        :param info: datastore profile
        :returns: None
        """
        info.project = info.project or config.default_project
        profile = self._query(
            session, DatastoreProfile, name=info.name, project=info.project
        ).one_or_none()
        if profile:
            profile.type = info.type
            profile.full_object = info.object
            self._commit(session, [profile])
        else:
            profile = DatastoreProfile(
                name=info.name,
                type=info.type,
                project=info.project,
                full_object=info.object,
            )
            self._upsert(session, [profile])

    def get_datastore_profile(
        self,
        session,
        profile: str,
        project: str,
    ):
        """
        get a datastore profile.
        :param session: SQLAlchemy session
        :param profile: name of the profile
        :param project: Name of the project
        :returns: None
        """
        project = project or config.default_project
        res = self._query(session, DatastoreProfile, name=profile, project=project)
        if res.first():
            return self._transform_datastore_profile_model_to_schema(res.first())
        else:
            raise mlrun.errors.MLRunNotFoundError(
                f"Datastore profile '{profile}' not found in project '{project}'"
            )

    def delete_datastore_profile(
        self,
        session,
        profile: str,
        project: str,
    ):
        project = project or config.default_project
        res = self._query(session, DatastoreProfile, name=profile, project=project)
        if res.first():
            session.delete(res.first())
            session.commit()
        else:
            raise mlrun.errors.MLRunNotFoundError(
                f"Datastore profile '{profile}' not found in project '{project}'"
            )

    def list_datastore_profiles(
        self,
        session,
        project: str,
    ):
        """
        list all datastore profiles for a project.
        :param session: SQLAlchemy session
        :param project: Name of the project
        :returns: List of DatatoreProfile objects (only the public portion of it)
        """
        project = project or config.default_project
        datastore_records = self._query(session, DatastoreProfile, project=project)
        return [
            self._transform_datastore_profile_model_to_schema(datastore_record)
            for datastore_record in datastore_records
        ]

    def delete_datastore_profiles(
        self,
        session,
        project: str,
    ):
        """
        Delete all datastore profiles.
        :param session: SQLAlchemy session
        :param project: Name of the project
        :returns: None
        """
        project = project or config.default_project
        query_results = self._query(session, DatastoreProfile, project=project)
        for profile in query_results:
            session.delete(profile)
        session.commit()

    @staticmethod
    def _transform_datastore_profile_model_to_schema(
        db_object,
    ) -> mlrun.common.schemas.DatastoreProfile:
        return mlrun.common.schemas.DatastoreProfile(
            name=db_object.name,
            type=db_object.type,
            object=db_object.full_object,
            project=db_object.project,
        )

    # --- Pagination ---
    def store_paginated_query_cache_record(
        self,
        session,
        user: str,
        function: str,
        current_page: int,
        page_size: int,
        kwargs: dict,
    ):
        # generate key hash from user, function, current_page and kwargs
        key = hashlib.sha256(
            f"{user}/{function}/{page_size}/{kwargs}".encode()
        ).hexdigest()
        existing_record = self.get_paginated_query_cache_record(session, key)
        if existing_record:
            existing_record.current_page = current_page
            existing_record.last_accessed = datetime.now(timezone.utc)
            param_record = existing_record
        else:
            param_record = PaginationCache(
                key=key,
                user=user,
                function=function,
                current_page=current_page,
                page_size=page_size,
                kwargs=kwargs,
                last_accessed=datetime.now(timezone.utc),
            )

        self._upsert(session, [param_record])
        return key

    def get_paginated_query_cache_record(
        self,
        session,
        key: str,
    ):
        return self._query(session, PaginationCache, key=key).one_or_none()

    def list_paginated_query_cache_record(
        self,
        session,
        key: str = None,
        user: str = None,
        function: str = None,
        last_accessed_before: datetime = None,
        order_by: typing.Optional[mlrun.common.schemas.OrderType] = None,
        as_query: bool = False,
    ):
        query = self._query(session, PaginationCache)
        if key:
            query = query.filter(PaginationCache.key == key)
        if user:
            query = query.filter(PaginationCache.user == user)
        if function:
            query = query.filter(PaginationCache.function == function)
        if last_accessed_before:
            query = query.filter(PaginationCache.last_accessed < last_accessed_before)

        if order_by:
            query = query.order_by(
                order_by.to_order_by_predicate(PaginationCache.last_accessed)
            )

        if as_query:
            return query

        return query.all()

    def delete_paginated_query_cache_record(
        self,
        session,
        key: str,
    ):
        self._delete(session, PaginationCache, key=key)

    def store_time_window_tracker_record(
        self,
        session: Session,
        key: str,
        timestamp: typing.Optional[datetime] = None,
        max_window_size_seconds: typing.Optional[int] = None,
    ) -> TimeWindowTracker:
        time_window_tracker_record = self.get_time_window_tracker_record(
            session, key=key, raise_on_not_found=False
        )
        if not time_window_tracker_record:
            time_window_tracker_record = TimeWindowTracker(key=key)

        if timestamp:
            time_window_tracker_record.timestamp = timestamp
        if max_window_size_seconds:
            time_window_tracker_record.max_window_size_seconds = max_window_size_seconds

        self._upsert(session, [time_window_tracker_record])
        return time_window_tracker_record

    def get_time_window_tracker_record(
        self, session, key: str, raise_on_not_found: bool = True
    ) -> TimeWindowTracker:
        time_window_tracker_record = self._query(
            session, TimeWindowTracker, key=key
        ).one_or_none()
        if not time_window_tracker_record and raise_on_not_found:
            raise mlrun.errors.MLRunNotFoundError(
                f"Time window tracker record not found: key={key}"
            )
        return time_window_tracker_record

    # ---- Utils ----
    def delete_table_records(
        self,
        session: Session,
        table: type[Base],
        raise_on_not_exists=True,
    ):
        """Delete all records from a table

        :param session: SQLAlchemy session
        :param table: the table class
        :param raise_on_not_exists: raise an error if the table does not exist
        """
        return self.delete_table_records_by_name(
            session, table.__tablename__, raise_on_not_exists
        )

    def delete_table_records_by_name(
        self,
        session: Session,
        table_name: str,
        raise_on_not_exists=True,
    ):
        """
        Delete a table by its name

        :param session: SQLAlchemy session
        :param table_name: table name
        :param raise_on_not_exists: raise an error if the table does not exist
        """

        # sanitize table name to prevent SQL injection, by removing all non-alphanumeric characters or underscores
        sanitized_table_name = re.sub(r"[^a-zA-Z0-9_]", "", table_name)

        # checking if the table exists can also help prevent SQL injection
        if self._is_table_exists(session, sanitized_table_name):
            truncate_statement = text(f"DELETE FROM {sanitized_table_name}")
            session.execute(truncate_statement)
            session.commit()
            return

        if raise_on_not_exists:
            raise mlrun.errors.MLRunNotFoundError(
                f"Table not found: {sanitized_table_name}"
            )
        logger.warning(
            "Table not found, skipping delete",
            table_name=sanitized_table_name,
        )

    @staticmethod
    def _is_table_exists(session: Session, table_name: str) -> bool:
        """
        Check if a table exists

        :param table_name: table name
        :return: True if the table exists, False otherwise
        """
        metadata = MetaData(bind=session.bind)
        metadata.reflect()
        return table_name in metadata.tables.keys()

    @staticmethod
    def _paginate_query(query, page: int = None, page_size: int = None):
        if page is not None:
            page_size = page_size or config.httpdb.pagination.default_page_size
            if query.count() < page_size * (page - 1):
                raise StopIteration
            query = query.limit(page_size).offset((page - 1) * page_size)

        return query
