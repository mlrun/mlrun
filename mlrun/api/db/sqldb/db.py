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
import asyncio
import collections
import functools
import re
import typing
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import fastapi.concurrency
import mergedeep
import pytz
from sqlalchemy import and_, distinct, func, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, aliased

import mlrun
import mlrun.api.db.session
import mlrun.api.utils.projects.remotes.follower
import mlrun.errors
from mlrun.api import schemas
from mlrun.api.db.base import DBInterface
from mlrun.api.db.sqldb.helpers import (
    generate_query_predicate_for_name,
    label_set,
    run_labels,
    run_start_time,
    run_state,
    update_labels,
)
from mlrun.api.db.sqldb.models import (
    Artifact,
    BackgroundTask,
    DataVersion,
    Entity,
    Feature,
    FeatureSet,
    FeatureVector,
    Function,
    Log,
    MarketplaceSource,
    Project,
    Run,
    Schedule,
    User,
    _labeled,
    _tagged,
)
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.lists import ArtifactList, FunctionList, RunList
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
    validate_tag_name,
)

NULL = None  # Avoid flake8 issuing warnings when comparing in filter
run_time_fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
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
    def __init__(self, dsn):
        self.dsn = dsn
        self._cache = {
            "project_resources_counters": {"value": None, "ttl": datetime.min}
        }
        self._name_with_iter_regex = re.compile("^[0-9]+-.+$")

    def initialize(self, session):
        pass

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
            "Storing run to db", project=project, uid=uid, iter=iter, run=run_data
        )
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
        run = self._get_run(session, uid, project, iter)
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
        update_labels(run, run_labels(struct))
        self._update_run_updated_time(run, struct)
        run.struct = struct
        self._upsert(session, [run])
        self._delete_empty_labels(session, Run.Label)

    def list_distinct_runs_uids(
        self,
        session,
        project: str = None,
        requested_logs_modes: typing.List[bool] = None,
        only_uids=True,
        last_start_time_from: datetime = None,
        states: typing.List[str] = None,
    ) -> typing.Union[typing.List[str], RunList]:
        """
        List all runs uids in the DB
        :param session: DB session
        :param project: Project name, `*` or `None` lists across all projects
        :param requested_logs_modes: If not `None`, will return only runs with the given requested logs modes
        :param only_uids: If True, will return only the uids of the runs as list of strings
                          If False, will return the full run objects as RunList
        :param last_start_time_from: If not `None`, will return only runs created after this time
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

        if last_start_time_from is not None:
            query = query.filter(Run.start_time >= last_start_time_from)

        if requested_logs_modes is not None:
            query = query.filter(Run.requested_logs.in_(requested_logs_modes))

        if not only_uids:
            # group_by allows us to have a row per uid with the whole record rather than just the uid (as distinct does)
            # note we cannot promise that the same row will be returned each time per uid as the order is not guaranteed
            query = query.group_by(Run.uid)

            runs = RunList()
            for run in query:
                runs.append(run.struct)

            return runs

        # from each row we expect to get a tuple of (uid,) so we need to extract the uid from the tuple
        return [uid for uid, in query.all()]

    def update_runs_requested_logs(
        self, session, uids: List[str], requested_logs: bool = True
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
            raise mlrun.errors.MLRunNotFoundError(f"Run {uid}:{project} not found")
        return run.struct

    def list_runs(
        self,
        session,
        name=None,
        uid: typing.Optional[typing.Union[str, List[str]]] = None,
        project=None,
        labels=None,
        states=None,
        sort=True,
        last=0,
        iter=False,
        start_time_from=None,
        start_time_to=None,
        last_update_time_from=None,
        last_update_time_to=None,
        partition_by: schemas.RunPartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: schemas.SortField = None,
        partition_order: schemas.OrderType = schemas.OrderType.desc,
        max_partitions: int = 0,
        requested_logs: bool = None,
        return_as_run_structs: bool = True,
    ):
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
        if partition_by:
            self._assert_partition_by_parameters(
                schemas.RunPartitionByField, partition_by, partition_sort_by
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
        if not return_as_run_structs:
            return query.all()

        runs = RunList()
        for run in query:
            runs.append(run.struct)

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

    @retry_on_conflict
    def overwrite_artifacts_with_tag(
        self,
        session: Session,
        project: str,
        tag: str,
        identifiers: typing.List[mlrun.api.schemas.ArtifactIdentifier],
    ):
        # query all artifacts which match the identifiers
        artifacts = []
        for identifier in identifiers:
            artifacts += self._list_artifacts_for_tagging(
                session,
                project_name=project,
                identifier=identifier,
            )
        # TODO remove duplicates artifacts entries
        # delete related tags from artifacts identifiers
        # not committing the session here because we want to do it atomic with the next query
        self._delete_artifacts_tags(session, project, artifacts, commit=False)
        # tag artifacts with tag
        self.tag_artifacts(session, artifacts, project, name=tag)

    @retry_on_conflict
    def append_tag_to_artifacts(
        self,
        session: Session,
        project: str,
        tag: str,
        identifiers: typing.List[mlrun.api.schemas.ArtifactIdentifier],
    ):
        # query all artifacts which match the identifiers
        artifacts = []
        for identifier in identifiers:
            artifacts += self._list_artifacts_for_tagging(
                session,
                project_name=project,
                identifier=identifier,
            )
        self.tag_artifacts(session, artifacts, project, name=tag)

    def delete_tag_from_artifacts(
        self,
        session: Session,
        project: str,
        tag: str,
        identifiers: typing.List[mlrun.api.schemas.ArtifactIdentifier],
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

    def _list_artifacts_for_tagging(
        self,
        session: Session,
        project_name: str,
        identifier: mlrun.api.schemas.ArtifactIdentifier,
    ):
        return self.list_artifacts(
            session,
            project=project_name,
            name=identifier.key,
            kind=identifier.kind,
            iter=identifier.iter,
            # 1. will be changed to uid, after refactoring the code, currently to list artifacts by uid
            # we are passing it into the tag param and resolve whether it's a uid or a tag in the
            # list_artifacts method (_resolve_tag)
            # 2. if the identifier.uid is None, we want to list all artifacts, so we pass "*"
            tag=identifier.uid or "*",
            as_records=True,
            # 1. because of backwards compatible that list_artifacts is keeping, we want to pass the function
            # indication that the tag which is passed is uid
            # 2. if uid wasn't passed as part of the identifiers then
            # we will ask for tag "*" and in that case we don't want to use the tag as uid
            use_tag_as_uid=True if identifier.uid else False,
        )

    @retry_on_conflict
    def store_artifact(
        self,
        session,
        key,
        artifact,
        uid,
        iter=None,
        tag="",
        project="",
    ):
        self._store_artifact(
            session,
            key,
            artifact,
            uid,
            iter,
            tag,
            project,
        )

    @staticmethod
    def _process_artifact_dict_to_store(artifact, key, iter=None):
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
    def _process_legacy_artifact_dict_to_store(artifact, key, iter=None):
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

    def _store_artifact(
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
        project = project or config.default_project
        artifact = deepcopy(artifact)
        if is_legacy_artifact(artifact):
            updated, key, labels = self._process_legacy_artifact_dict_to_store(
                artifact, key, iter
            )
        else:
            updated, key, labels = self._process_artifact_dict_to_store(
                artifact, key, iter
            )
        existed = True
        art = self._get_artifact(session, uid, project, key)
        if not art:
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
            self.tag_artifacts(session, [art], project, tag)
            # we want to tag the artifact also as "latest" if it's the first time we store it, reason is that there are
            # updates we are doing to the metadata of the artifact (like updating the labels) and we don't want those
            # changes to be reflected in the "latest" tag, as this in not actual the "latest" version of the artifact
            # which was produced by the user
            if not existed and tag != "latest":
                self.tag_artifacts(session, [art], project, "latest")

    @staticmethod
    def _set_tag_in_artifact_struct(artifact, tag):
        if is_legacy_artifact(artifact):
            artifact["tag"] = tag
        else:
            artifact["metadata"]["tag"] = tag

    def _add_tags_to_artifact_struct(
        self, session, artifact_struct, artifact_id, tag=None
    ):
        artifacts = []
        if tag and tag != "*":
            self._set_tag_in_artifact_struct(artifact_struct, tag)
            artifacts.append(artifact_struct)
        else:
            tag_results = self._query(session, Artifact.Tag, obj_id=artifact_id).all()
            if not tag_results:
                return [artifact_struct]
            for tag_object in tag_results:
                artifact_with_tag = deepcopy(artifact_struct)
                self._set_tag_in_artifact_struct(artifact_with_tag, tag_object.name)
                artifacts.append(artifact_with_tag)
        return artifacts

    def read_artifact(self, session, key, tag="", iter=None, project=""):
        project = project or config.default_project
        ids = self._resolve_tag(session, Artifact, project, tag)
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
        iter: int = None,
        best_iteration: bool = False,
        as_records: bool = False,
        use_tag_as_uid: bool = None,
    ):
        project = project or config.default_project

        if best_iteration and iter is not None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "best-iteration cannot be used when iter is specified"
            )
        # TODO: Refactor this area
        ids = "*"
        if tag:
            # use_tag_as_uid is used to catch old artifacts which were created when logging artifacts using the project
            # producer and not by context, this because when were logging artifacts using the project producer we were
            # setting the artifact uid to be the same as the producer tag which at the time was "latest", this becomes a
            # problem with uid="latest" because there are also "latest" tags in the system, which means we will get ids
            # response from the `_resolve_tag` above and then we will iterate over the wrong artifact
            # use_tag_as_uid==None is keeping the old behavior
            # use_tag_as_uid==False also keeps the old behavior for now, but left that option to be able to change later
            # use_tag_as_uid==True saying to the list artifacts that the tag is actually the uid
            if tag == "*" or use_tag_as_uid:
                ids = tag
            else:
                ids = self._resolve_tag(session, Artifact, project, tag)

        artifacts = ArtifactList()
        artifact_records = self._find_artifacts(
            session,
            project,
            ids,
            labels,
            since,
            until,
            name,
            kind,
            category,
            iter,
            use_tag_as_uid=use_tag_as_uid,
        )
        if as_records:
            if best_iteration:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "as_records is not supported with best_iteration=True"
                )
            return artifact_records

        # In case best_iteration is requested and filtering is done by tag, then we might have artifacts from the best
        # iteration (which is tagged) but not the link artifact (that is not tagged). This will cause the filtering
        # happening next to fail. Therefore, we need to add to the list of results the link artifacts from the given
        # keys + uids.
        if best_iteration:
            iteration_keys = set()
            link_keys = set()
            for artifact in artifact_records:
                if self._name_with_iter_regex.match(artifact.key):
                    key_without_iteration = artifact.key.split("-", maxsplit=1)[1]
                    iteration_keys.add((key_without_iteration, artifact.uid))
                else:
                    link_keys.add((artifact.key, artifact.uid))
            missing_link_keys = iteration_keys.difference(link_keys)
            artifact_records.extend(
                self._get_link_artifacts_by_keys_and_uids(
                    session, project, missing_link_keys
                )
            )

        # concatenating <artifact.key> and <artifact.uid> to create a unique key for the artifacts
        indexed_artifacts = {
            f"{artifact.key}-{artifact.uid}": artifact for artifact in artifact_records
        }
        for artifact in artifact_records:
            has_iteration = self._name_with_iter_regex.match(artifact.key)

            # Handle case of looking for best iteration. In that case if there's a linked iteration, we look
            # for it in the results and return the linked artifact if exists. If original iter is not 0 then
            # we skip this item.
            if best_iteration:
                if has_iteration:
                    continue
                if is_legacy_artifact(artifact.struct):
                    link_iteration = artifact.struct.get("link_iteration")
                else:
                    link_iteration = artifact.struct["spec"].get("link_iteration")

                if link_iteration:
                    # link artifact key is without the iteration so to pull the linked artifact we need to
                    # concatenate the <link-iteration>-<artifact.key>-<artifact.uid> together
                    linked_key = f"{link_iteration}-{artifact.key}-{artifact.uid}"
                    linked_artifact = indexed_artifacts.get(linked_key)
                    if linked_artifact:
                        artifact = linked_artifact
                    else:
                        continue

            # We need special handling for the case where iter==0, since in that case no iter prefix will exist.
            # Regex support is db-specific, and SQLAlchemy actually implements Python regex for SQLite anyway,
            # and even that only in SA 1.4. So doing this here rather than in the query.
            if iter == 0 and has_iteration:
                continue

            artifact_struct = artifact.struct

            # set the tags in the artifact struct
            artifacts_with_tag = self._add_tags_to_artifact_struct(
                session, artifact_struct, artifact.id, tag
            )
            artifacts.extend(artifacts_with_tag)

        return artifacts

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

    def del_artifact(self, session, key, tag="", project=""):
        project = project or config.default_project

        query = session.query(Artifact).filter(
            Artifact.key == key, Artifact.project == project
        )
        if tag:
            query = query.join(Artifact.Tag).filter(Artifact.Tag.name == tag)

        # Cannot delete yet, because tag and label deletion queries join with the artifacts table, so the objects
        # still need to be there.
        artifacts = query.all()
        if not artifacts:
            return

        # deleting tags and labels, because in sqlite the relationships aren't necessarily cascading
        self._delete_artifact_tags(session, project, key, tag, commit=False)
        self._delete_class_labels(
            session, Artifact, project=project, key=key, commit=False
        )
        for artifact in artifacts:
            session.delete(artifact)
        session.commit()

    def _delete_artifacts_tags(
        self,
        session,
        project: str,
        artifacts: typing.List[Artifact],
        tags: typing.List[str] = None,
        commit: bool = True,
    ):
        artifacts_keys = [str(artifact.key) for artifact in artifacts]
        query = (
            session.query(Artifact.Tag)
            .join(Artifact)
            .filter(
                Artifact.project == project,
                Artifact.key.in_(artifacts_keys),
            )
        )
        if tags:
            query = query.filter(Artifact.Tag.name.in_(tags))
        for tag in query:
            session.delete(tag)
        if commit:
            session.commit()

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
        ids = "*"
        if tag and tag != "*":
            ids = self._resolve_tag(session, Artifact, project, tag)
        distinct_keys = {
            artifact.key
            for artifact in self._find_artifacts(
                session, project, ids, labels, name=name
            )
        }
        for key in distinct_keys:
            self.del_artifact(session, key, "", project)

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
            function=function,
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
                "Conflict between requested name and name in function body"
            )
        if not body_name:
            function.setdefault("metadata", {})["name"] = name
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

            # If queried by hash key and nuclio/serving function remove status
            is_nuclio = (
                function.get("kind", "")
                in mlrun.runtimes.RuntimeKinds.nuclio_runtimes()
            )
            if hash_key and is_nuclio:
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

        # deleting tags and labels, because in sqlite the relationships aren't necessarily cascading
        self._delete_function_tags(session, project, name, commit=False)
        self._delete_class_labels(
            session, Function, project=project, name=name, commit=False
        )
        self._delete(session, Function, project=project, name=name)

    def _delete_functions(self, session: Session, project: str):
        for function_name in self._list_project_function_names(session, project):
            self.delete_function(session, project, function_name)

    def _list_project_function_names(
        self, session: Session, project: str
    ) -> typing.List[str]:
        return [
            name
            for name, in self._query(
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

    def list_artifact_tags(
        self, session, project, category: schemas.ArtifactCategories = None
    ) -> typing.List[typing.Tuple[str, str, str]]:
        """
        :return: a list of Tuple of (project, artifact.key, tag)
        """
        # TODO - refactor once we have the artifact kind as a field in the DB, the filtering on category can be done
        # as a simple SQL query, and don't need to use the extra processing of listing tags etc.

        artifacts = self.list_artifacts(
            session, project=project, tag="*", category=category
        )
        results = []
        for artifact in artifacts:
            if is_legacy_artifact(artifact):
                results.append((project, artifact.get("db_key"), artifact.get("tag")))
            else:
                results.append(
                    (
                        project,
                        artifact["spec"].get("db_key"),
                        artifact["metadata"].get("tag"),
                    )
                )

        return results

    def create_schedule(
        self,
        session: Session,
        project: str,
        name: str,
        kind: schemas.ScheduleKinds,
        scheduled_object: Any,
        cron_trigger: schemas.ScheduleCronTrigger,
        concurrency_limit: int,
        labels: Dict = None,
        next_run_time: datetime = None,
    ):
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

        labels = labels or {}
        update_labels(schedule, labels)

        logger.debug(
            "Saving schedule to db",
            project=project,
            name=name,
            kind=kind,
            cron_trigger=cron_trigger,
            concurrency_limit=concurrency_limit,
            next_run_time=next_run_time,
        )
        self._upsert(session, [schedule])

    def update_schedule(
        self,
        session: Session,
        project: str,
        name: str,
        scheduled_object: Any = None,
        cron_trigger: schemas.ScheduleCronTrigger = None,
        labels: Dict = None,
        last_run_uri: str = None,
        concurrency_limit: int = None,
        next_run_time: datetime = None,
    ):
        schedule = self._get_schedule_record(session, project, name)

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
            query = query.filter(generate_query_predicate_for_name(Schedule.name, name))
        labels = label_set(labels)
        query = self._add_labels_filter(session, query, Schedule, labels)

        schedules = [
            self._transform_schedule_record_to_scheme(db_schedule)
            for db_schedule in query
        ]
        return schedules

    def get_schedule(
        self, session: Session, project: str, name: str
    ) -> schemas.ScheduleRecord:
        logger.debug("Getting schedule from db", project=project, name=name)
        schedule_record = self._get_schedule_record(session, project, name)
        schedule = self._transform_schedule_record_to_scheme(schedule_record)
        return schedule

    def _get_schedule_record(
        self, session: Session, project: str, name: str
    ) -> schemas.ScheduleRecord:
        query = self._query(session, Schedule, project=project, name=name)
        schedule_record = query.one_or_none()
        if not schedule_record:
            raise mlrun.errors.MLRunNotFoundError(
                f"Schedule not found: project={project}, name={name}"
            )
        return schedule_record

    def delete_schedule(self, session: Session, project: str, name: str):
        logger.debug("Removing schedule from db", project=project, name=name)
        self._delete_class_labels(
            session, Schedule, project=project, name=name, commit=False
        )
        self._delete(session, Schedule, project=project, name=name)

    def delete_schedules(self, session: Session, project: str):
        logger.debug("Removing schedules from db", project=project)
        for schedule in self.list_schedules(session, project=project):
            self.delete_schedule(session, project, schedule.name)

    def _delete_feature_sets(self, session: Session, project: str):
        logger.debug("Removing feature-sets from db", project=project)
        for feature_set_name in self._list_project_feature_set_names(session, project):
            self.delete_feature_set(session, project, feature_set_name)

    def _list_project_feature_set_names(
        self, session: Session, project: str
    ) -> typing.List[str]:
        return [
            name
            for name, in self._query(
                session, distinct(FeatureSet.name), project=project
            ).all()
        ]

    def _delete_feature_vectors(self, session: Session, project: str):
        logger.debug("Removing feature-vectors from db", project=project)
        for feature_vector_name in self._list_project_feature_vector_names(
            session, project
        ):
            self.delete_feature_vector(session, project, feature_vector_name)

    def _list_project_feature_vector_names(
        self, session: Session, project: str
    ) -> typing.List[str]:
        return [
            name
            for name, in self._query(
                session, distinct(FeatureVector.name), project=project
            ).all()
        ]

    def tag_artifacts(self, session, artifacts, project: str, name: str):
        # found a bug in here, which is being exposed for when have multi-param execution, this because each
        # artifact key is being concatenated with the key and the iteration, this because problemtic in this query
        # because we are filtering by the key+iteration and not just the key ( which would require some regex )
        # this would be fixed as part of the refactoring of the new artifact table structure where we would have
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

    def tag_objects_v2(self, session, objs, project: str, name: str):
        tags = []
        for obj in objs:
            query = self._query(
                session, obj.Tag, name=name, project=project, obj_name=obj.name
            )
            tag = query.one_or_none()
            if not tag:
                tag = obj.Tag(project=project, name=name, obj_name=obj.name)
            tag.obj_id = obj.id
            tags.append(tag)
        self._upsert(session, tags)

    def create_project(self, session: Session, project: schemas.Project):
        logger.debug("Creating project in DB", project=project)
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
            full_object=project.dict(),
        )
        labels = project.metadata.labels or {}
        update_labels(project_record, labels)
        self._upsert(session, [project_record])

    @retry_on_conflict
    def store_project(self, session: Session, name: str, project: schemas.Project):
        logger.debug("Storing project in DB", name=name, project=project)
        project_record = self._get_project_record(
            session, name, raise_on_not_found=False
        )
        if not project_record:
            self.create_project(session, project)
        else:
            self._update_project_record_from_project(session, project_record, project)

    def patch_project(
        self,
        session: Session,
        name: str,
        project: dict,
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
    ):
        logger.debug(
            "Patching project in DB", name=name, project=project, patch_mode=patch_mode
        )
        project_record = self._get_project_record(session, name)
        self._patch_project_record_from_project(
            session, name, project_record, project, patch_mode
        )

    def get_project(
        self, session: Session, name: str = None, project_id: int = None
    ) -> schemas.Project:
        project_record = self._get_project_record(session, name, project_id)

        return self._transform_project_record_to_schema(session, project_record)

    def delete_project(
        self,
        session: Session,
        name: str,
        deletion_strategy: schemas.DeletionStrategy = schemas.DeletionStrategy.default(),
    ):
        logger.debug(
            "Deleting project from DB", name=name, deletion_strategy=deletion_strategy
        )
        self._delete(session, Project, name=name)

    def list_projects(
        self,
        session: Session,
        owner: str = None,
        format_: mlrun.api.schemas.ProjectsFormat = mlrun.api.schemas.ProjectsFormat.full,
        labels: List[str] = None,
        state: mlrun.api.schemas.ProjectState = None,
        names: typing.Optional[typing.List[str]] = None,
    ) -> schemas.ProjectsOutput:
        query = self._query(session, Project, owner=owner, state=state)
        if labels:
            query = self._add_labels_filter(session, query, Project, labels)
        if names is not None:
            query = query.filter(Project.name.in_(names))
        project_records = query.all()
        projects = []
        for project_record in project_records:
            if format_ == mlrun.api.schemas.ProjectsFormat.name_only:
                projects = [project_record.name for project_record in project_records]
            # leader format is only for follower mode which will format the projects returned from here
            elif format_ in [
                mlrun.api.schemas.ProjectsFormat.full,
                mlrun.api.schemas.ProjectsFormat.leader,
            ]:
                projects.append(
                    self._transform_project_record_to_schema(session, project_record)
                )
            else:
                raise NotImplementedError(
                    f"Provided format is not supported. format={format_}"
                )
        return schemas.ProjectsOutput(projects=projects)

    async def get_project_resources_counters(
        self,
    ) -> Tuple[
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
    ]:
        results = await asyncio.gather(
            fastapi.concurrency.run_in_threadpool(
                mlrun.api.db.session.run_function_with_new_db_session,
                self._calculate_files_counters,
            ),
            fastapi.concurrency.run_in_threadpool(
                mlrun.api.db.session.run_function_with_new_db_session,
                self._calculate_schedules_counters,
            ),
            fastapi.concurrency.run_in_threadpool(
                mlrun.api.db.session.run_function_with_new_db_session,
                self._calculate_feature_sets_counters,
            ),
            fastapi.concurrency.run_in_threadpool(
                mlrun.api.db.session.run_function_with_new_db_session,
                self._calculate_models_counters,
            ),
            fastapi.concurrency.run_in_threadpool(
                mlrun.api.db.session.run_function_with_new_db_session,
                self._calculate_runs_counters,
            ),
        )
        (
            project_to_files_count,
            project_to_schedule_count,
            project_to_feature_set_count,
            project_to_models_count,
            (
                project_to_recent_failed_runs_count,
                project_to_running_runs_count,
            ),
        ) = results
        return (
            project_to_files_count,
            project_to_schedule_count,
            project_to_feature_set_count,
            project_to_models_count,
            project_to_recent_failed_runs_count,
            project_to_running_runs_count,
        )

    def _calculate_functions_counters(self, session) -> Dict[str, int]:
        functions_count_per_project = (
            session.query(Function.project, func.count(distinct(Function.name)))
            .group_by(Function.project)
            .all()
        )
        project_to_function_count = {
            result[0]: result[1] for result in functions_count_per_project
        }
        return project_to_function_count

    def _calculate_schedules_counters(self, session) -> Dict[str, int]:
        schedules_count_per_project = (
            session.query(Schedule.project, func.count(distinct(Schedule.name)))
            .group_by(Schedule.project)
            .all()
        )
        project_to_schedule_count = {
            result[0]: result[1] for result in schedules_count_per_project
        }
        return project_to_schedule_count

    def _calculate_feature_sets_counters(self, session) -> Dict[str, int]:
        feature_sets_count_per_project = (
            session.query(FeatureSet.project, func.count(distinct(FeatureSet.name)))
            .group_by(FeatureSet.project)
            .all()
        )
        project_to_feature_set_count = {
            result[0]: result[1] for result in feature_sets_count_per_project
        }
        return project_to_feature_set_count

    def _calculate_models_counters(self, session) -> Dict[str, int]:
        import mlrun.artifacts

        # The kind filter is applied post the query to the DB (manually in python code), so counting should be that
        # way as well, therefore we're doing it here, and can't do it with sql as the above
        # We're using the "latest" which gives us only one version of each artifact key, which is what we want to
        # count (artifact count, not artifact versions count)
        model_artifacts = self._find_artifacts(
            session, None, "latest", kind=mlrun.artifacts.model.ModelArtifact.kind
        )
        project_to_models_count = collections.defaultdict(int)
        for model_artifact in model_artifacts:
            project_to_models_count[model_artifact.project] += 1
        return project_to_models_count

    def _calculate_files_counters(self, session) -> Dict[str, int]:
        import mlrun.artifacts

        # The category filter is applied post the query to the DB (manually in python code), so counting should be that
        # way as well, therefore we're doing it here, and can't do it with sql as the above
        # We're using the "latest" which gives us only one version of each artifact key, which is what we want to
        # count (artifact count, not artifact versions count)
        file_artifacts = self._find_artifacts(
            session, None, "latest", category=mlrun.api.schemas.ArtifactCategories.other
        )
        project_to_files_count = collections.defaultdict(int)
        for file_artifact in file_artifacts:
            project_to_files_count[file_artifact.project] += 1
        return project_to_files_count

    def _calculate_runs_counters(
        self, session
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        running_runs_count_per_project = (
            session.query(Run.project, func.count(distinct(Run.name)))
            .filter(
                Run.state.in_(mlrun.runtimes.constants.RunStates.non_terminal_states())
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
                        mlrun.runtimes.constants.RunStates.error,
                        mlrun.runtimes.constants.RunStates.aborted,
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
        return project_to_recent_failed_runs_count, project_to_running_runs_count

    async def generate_projects_summaries(
        self, session: Session, projects: List[str]
    ) -> List[mlrun.api.schemas.ProjectSummary]:
        (
            project_to_function_count,
            project_to_schedule_count,
            project_to_feature_set_count,
            project_to_models_count,
            project_to_recent_failed_runs_count,
            project_to_running_runs_count,
        ) = await self._get_project_resources_counters(session)
        project_summaries = []
        for project in projects:
            project_summaries.append(
                mlrun.api.schemas.ProjectSummary(
                    name=project,
                    functions_count=project_to_function_count.get(project, 0),
                    schedules_count=project_to_schedule_count.get(project, 0),
                    feature_sets_count=project_to_feature_set_count.get(project, 0),
                    models_count=project_to_models_count.get(project, 0),
                    runs_failed_recent_count=project_to_recent_failed_runs_count.get(
                        project, 0
                    ),
                    runs_running_count=project_to_running_runs_count.get(project, 0),
                    # This is a mandatory field - filling here with 0, it will be filled with the real number in the
                    # crud layer
                    pipelines_running_count=0,
                )
            )
        return project_summaries

    def _update_project_record_from_project(
        self, session: Session, project_record: Project, project: schemas.Project
    ):
        project.metadata.created = project_record.created
        project_dict = project.dict()
        # TODO: handle taking out the functions/workflows/artifacts out of the project and save them separately
        project_record.full_object = project_dict
        project_record.description = project.spec.description
        project_record.source = project.spec.source
        project_record.owner = project.spec.owner
        project_record.state = project.status.state
        labels = project.metadata.labels or {}
        update_labels(project_record, labels)
        self._upsert(session, [project_record])

    def _patch_project_record_from_project(
        self,
        session: Session,
        name: str,
        project_record: Project,
        project: dict,
        patch_mode: schemas.PatchMode,
    ):
        project.setdefault("metadata", {})["created"] = project_record.created
        strategy = patch_mode.to_mergedeep_strategy()
        project_record_full_object = project_record.full_object
        mergedeep.merge(project_record_full_object, project, strategy=strategy)

        # If a bad kind value was passed, it will fail here (return 422 to caller)
        project = schemas.Project(**project_record_full_object)
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
    ) -> Project:
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
        self.del_runs(session, project=name)
        self.delete_schedules(session, name)
        self._delete_functions(session, name)
        self._delete_feature_sets(session, name)
        self._delete_feature_vectors(session, name)
        self._delete_background_tasks(session, project=name)

        # resources deletion should remove their tags and labels as well, but doing another try in case there are
        # orphan resources
        self._delete_resources_tags(session, name)
        self._delete_resources_labels(session, name)

    @staticmethod
    def _verify_empty_list_of_project_related_resources(
        project: str, resources: List, resource_name: str
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
    ):
        query = self._query(session, cls, name=name, project=project)
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

    def get_feature_set(
        self,
        session,
        project: str,
        name: str,
        tag: str = None,
        uid: str = None,
    ) -> schemas.FeatureSet:
        feature_set = self._get_feature_set(session, project, name, tag, uid)
        if not feature_set:
            feature_set_uri = generate_object_uri(project, name, tag)
            raise mlrun.errors.MLRunNotFoundError(
                f"Feature-set not found {feature_set_uri}"
            )

        return feature_set

    def _get_records_to_tags_map(self, session, cls, project, tag, name=None):
        # Find object IDs by tag, project and feature-set-name (which is a like query)
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
        self, object_record, transform_fn, obj_id_tags, default_tag=None
    ):
        # Using a similar mechanism here to assign tags to feature sets as is used in list_functions. Please refer
        # there for some comments explaining the logic.
        results = []
        if default_tag:
            results.append(transform_fn(object_record, default_tag))
        else:
            object_tags = obj_id_tags.get(object_record.id, [])
            if len(object_tags) == 0 and not object_record.uid.startswith(
                unversioned_tagged_object_uid_prefix
            ):
                new_object = transform_fn(object_record)
                results.append(new_object)
            else:
                for object_tag in object_tags:
                    results.append(transform_fn(object_record, object_tag))
        return results

    @staticmethod
    def _generate_feature_set_digest(feature_set: schemas.FeatureSet):
        return schemas.FeatureSetDigestOutput(
            metadata=feature_set.metadata,
            spec=schemas.FeatureSetDigestSpec(
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
        labels: List[str] = None,
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
        entities: List[str] = None,
        labels: List[str] = None,
    ) -> schemas.FeaturesOutput:
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
        for row in query:
            feature_record = schemas.FeatureRecord.from_orm(row.Feature)
            feature_name = feature_record.name

            feature_sets = self._generate_records_with_tags_assigned(
                row.FeatureSet,
                self._transform_feature_set_model_to_schema,
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

                features_results.append(
                    schemas.FeatureListOutput(
                        feature=feature,
                        feature_set_digest=self._generate_feature_set_digest(
                            feature_set
                        ),
                    )
                )
        return schemas.FeaturesOutput(features=features_results)

    def list_entities(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        labels: List[str] = None,
    ) -> schemas.EntitiesOutput:
        feature_set_id_tags = self._get_records_to_tags_map(
            session, FeatureSet, project, tag, name=None
        )

        query = self._generate_feature_or_entity_list_query(
            session, Entity, project, feature_set_id_tags.keys(), name, tag, labels
        )

        entities_results = []
        for row in query:
            entity_record = schemas.FeatureRecord.from_orm(row.Entity)
            entity_name = entity_record.name

            feature_sets = self._generate_records_with_tags_assigned(
                row.FeatureSet,
                self._transform_feature_set_model_to_schema,
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

                entities_results.append(
                    schemas.EntityListOutput(
                        entity=entity,
                        feature_set_digest=self._generate_feature_set_digest(
                            feature_set
                        ),
                    )
                )
        return schemas.EntitiesOutput(entities=entities_results)

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
            schemas.FeatureStorePartitionByField, schemas.RunPartitionByField
        ],
        rows_per_partition: int,
        partition_sort_by: schemas.SortField,
        partition_order: schemas.OrderType,
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
        subquery = query.add_column(row_number_column).subquery()

        if max_partitions == 0:
            # If we don't query on max-partitions, we end here. Need to alias the subquery so that the ORM will
            # be able to properly map it to objects.
            result_query = session.query(aliased(cls, subquery)).filter(
                subquery.c.row_number <= rows_per_partition
            )
            return result_query

        # Otherwise no need for an alias, as this is an internal query and will be wrapped by another one where
        # alias will apply. We just apply the filter here.
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
        result_query = result_query.add_column(partition_rank).subquery()
        result_query = session.query(aliased(cls, result_query)).filter(
            result_query.c.partition_rank <= max_partitions
        )
        return result_query

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
        partition_by: schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: schemas.SortField = None,
        partition_order: schemas.OrderType = schemas.OrderType.desc,
    ) -> schemas.FeatureSetsOutput:
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
                schemas.FeatureStorePartitionByField, partition_by, partition_sort_by
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
                )
            )
        return schemas.FeatureSetsOutput(feature_sets=feature_sets)

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
        feature_set: FeatureSet, feature_dicts: List[dict]
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
            if feature_dict["name"] in features_to_add:
                labels = feature_dict.get("labels") or {}
                feature = Feature(
                    name=feature_dict["name"],
                    value_type=feature_dict["value_type"],
                    labels=[],
                )
                update_labels(feature, labels)
                feature_set.features.append(feature)

    @staticmethod
    def _update_feature_set_entities(feature_set: FeatureSet, entity_dicts: List[dict]):
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

        labels = common_object_dict["metadata"].pop("labels", {}) or {}
        update_labels(db_object, labels)

    @retry_on_conflict
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
    ) -> str:
        original_uid = uid

        # record with the given tag/uid
        _, _, existing_feature_set = self._get_record_by_name_tag_and_uid(
            session, FeatureSet, project, name, tag, uid
        )

        feature_set_dict = feature_set.dict(exclude_none=True)

        # get the computed uid
        uid = self._common_object_validate_and_perform_uid_change(
            feature_set_dict, tag, versioned, original_uid
        )

        if existing_feature_set:
            if uid == existing_feature_set.uid or always_overwrite:
                db_feature_set = existing_feature_set
            else:

                # In case an object with the given tag (or 'latest' which is the default) and name, but different uid
                # was found - Check If an object with the same computed uid but different tag already exists
                # and re-tag it.
                if self._re_tag_existing_object(
                    session, FeatureSet, project, name, tag, uid
                ):
                    return uid

                db_feature_set = FeatureSet(project=project)

            self._update_db_record_from_object_dict(
                db_feature_set, feature_set_dict, uid
            )

            self._update_feature_set_spec(db_feature_set, feature_set_dict)
            self._upsert(session, [db_feature_set])
            if tag:
                self.tag_objects_v2(session, [db_feature_set], project, tag)
            return uid

        # Object with the given tag/uid doesn't exist
        # Check if this is a re-tag of existing object - search by uid only
        if self._re_tag_existing_object(session, FeatureSet, project, name, tag, uid):
            return uid

        feature_set.metadata.tag = tag
        return self.create_feature_set(session, project, feature_set, versioned)

    def _re_tag_existing_object(self, session, cls, project, name, tag, uid):
        _, _, existing_object = self._get_record_by_name_tag_and_uid(
            session, cls, project, name, None, uid
        )
        if existing_object:
            self.tag_objects_v2(session, [existing_object], project, tag)
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

    def create_feature_set(
        self,
        session,
        project,
        feature_set: schemas.FeatureSet,
        versioned=True,
    ) -> str:
        (uid, tag, feature_set_dict,) = self._validate_and_enrich_record_for_creation(
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
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
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

    def _delete_feature_store_object(self, session, cls, project, name, tag, uid):
        if tag and uid:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Both uid and tag specified when deleting an object."
            )

        object_id = None
        if uid:
            object_record = self._query(
                session, cls, project=project, name=name, uid=uid
            ).one_or_none()
            if object_record is None:
                return
            object_id = object_record.id
        elif tag:
            tag_record = self._query(
                session, cls.Tag, project=project, name=tag, obj_name=name
            ).one_or_none()
            if tag_record is None:
                return
            object_id = tag_record.obj_id

        if object_id:
            # deleting tags, because in sqlite the relationships aren't necessarily cascading
            self._delete(session, cls.Tag, obj_id=object_id)
            self._delete(session, cls, id=object_id)
        else:
            # If we got here, neither tag nor uid were provided - delete all references by name.
            # deleting tags, because in sqlite the relationships aren't necessarily cascading
            self._delete(session, cls.Tag, project=project, obj_name=name)
            self._delete(session, cls, project=project, name=name)

    def delete_feature_set(self, session, project, name, tag=None, uid=None):
        self._delete_feature_store_object(session, FeatureSet, project, name, tag, uid)

    def create_feature_vector(
        self,
        session,
        project,
        feature_vector: schemas.FeatureVector,
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

    def get_feature_vector(
        self, session, project: str, name: str, tag: str = None, uid: str = None
    ) -> schemas.FeatureVector:
        feature_vector = self._get_feature_vector(session, project, name, tag, uid)
        if not feature_vector:
            feature_vector_uri = generate_object_uri(project, name, tag)
            raise mlrun.errors.MLRunNotFoundError(
                f"Feature-vector not found {feature_vector_uri}"
            )

        return feature_vector

    def list_feature_vectors(
        self,
        session,
        project: str,
        name: str = None,
        tag: str = None,
        state: str = None,
        labels: List[str] = None,
        partition_by: schemas.FeatureStorePartitionByField = None,
        rows_per_partition: int = 1,
        partition_sort_by: schemas.SortField = None,
        partition_order: schemas.OrderType = schemas.OrderType.desc,
    ) -> schemas.FeatureVectorsOutput:
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
                schemas.FeatureStorePartitionByField, partition_by, partition_sort_by
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
        return schemas.FeatureVectorsOutput(feature_vectors=feature_vectors)

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
        feature_vector: schemas.FeatureVector,
        tag=None,
        uid=None,
        versioned=True,
        always_overwrite=False,
    ) -> str:
        original_uid = uid

        # record with the given tag/uid
        _, _, existing_feature_vector = self._get_record_by_name_tag_and_uid(
            session, FeatureVector, project, name, tag, uid
        )

        feature_vector_dict = feature_vector.dict(exclude_none=True)

        # get the computed uid
        uid = self._common_object_validate_and_perform_uid_change(
            feature_vector_dict, tag, versioned, original_uid
        )

        if existing_feature_vector:
            if uid == existing_feature_vector.uid or always_overwrite:
                db_feature_vector = existing_feature_vector
            else:

                # In case an object with the given tag (or 'latest' which is the default) and name, but different uid
                # was found - Check If an object with the same computed uid but different tag already exists
                # and re-tag it.
                if self._re_tag_existing_object(
                    session, FeatureVector, project, name, tag, uid
                ):
                    return uid

                db_feature_vector = FeatureVector(project=project)

            self._update_db_record_from_object_dict(
                db_feature_vector, feature_vector_dict, uid
            )

            self._upsert(session, [db_feature_vector])
            if tag:
                self.tag_objects_v2(session, [db_feature_vector], project, tag)

            return uid

        # Object with the given tag/uid doesn't exist
        # Check if this is a re-tag of existing object - search by uid only
        if self._re_tag_existing_object(
            session, FeatureVector, project, name, tag, uid
        ):
            return uid

        feature_vector.metadata.tag = tag
        return self.create_feature_vector(session, project, feature_vector, versioned)

    def patch_feature_vector(
        self,
        session,
        project,
        name,
        feature_vector_update: dict,
        tag=None,
        uid=None,
        patch_mode: schemas.PatchMode = schemas.PatchMode.replace,
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

        feature_vector = schemas.FeatureVector(**feature_vector_struct)
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
        self._delete_feature_store_object(
            session, FeatureVector, project, name, tag, uid
        )

    def _resolve_tag(self, session, cls, project, name):
        ids = []
        for tag in self._query(session, cls.Tag, project=project, name=name):
            ids.append(tag.obj_id)
        if not ids:
            return name  # Not found, return original uid
        return ids

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
            query = query.filter(
                generate_query_predicate_for_name(cls.Tag.obj_name, obj_name)
            )

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
                    f"add user: {err_to_str(err)}"
                ) from err
        return users

    def _get_class_instance_by_uid(self, session, cls, name, project, uid):
        query = self._query(session, cls, name=name, project=project, uid=uid)
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

    def _upsert(self, session, objects, ignore=False):
        if not objects:
            return
        for object_ in objects:
            session.add(object_)
        self._commit(session, objects, ignore)

    def _commit(self, session, objects, ignore=False):
        def _try_commit_obj():
            try:
                session.commit()
            except SQLAlchemyError as err:
                session.rollback()
                cls = objects[0].__class__.__name__
                if "database is locked" in str(err):
                    logger.warning(
                        "Database is locked. Retrying", cls=cls, err=str(err)
                    )
                    raise mlrun.errors.MLRunRuntimeError(
                        "Failed committing changes, database is locked"
                    ) from err
                logger.warning(
                    "Failed committing changes to DB", cls=cls, err=err_to_str(err)
                )
                if not ignore:
                    identifiers = ",".join(
                        object_.get_identifier_string() for object_ in objects
                    )
                    # We want to retry only when database is locked so for any other scenario escalate to fatal failure
                    try:
                        if any([message in str(err) for message in conflict_messages]):
                            raise mlrun.errors.MLRunConflictError(
                                f"Conflict - {cls} already exists: {identifiers}"
                            ) from err
                        raise mlrun.errors.MLRunRuntimeError(
                            f"Failed committing changes to DB. class={cls} objects={identifiers}"
                        ) from err
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

    def _latest_uid_filter(self, session, query):
        # Create a sub query of latest uid (by updated) per (project,key)
        subq = (
            session.query(
                Artifact.uid,
                Artifact.project,
                Artifact.key,
                func.max(Artifact.updated),
            )
            .group_by(
                Artifact.project,
                Artifact.key.label("key"),
            )
            .subquery("max_key")
        )

        # Join current query with sub query on (project, key, uid)
        return query.join(
            subq,
            and_(
                Artifact.project == subq.c.project,
                Artifact.key == subq.c.key,
                Artifact.uid == subq.c.uid,
            ),
        )

    @staticmethod
    def _escape_characters_for_like_query(value: str) -> str:
        return (
            value.translate(value.maketrans({"_": r"\_", "%": r"\%"})) if value else ""
        )

    def _add_artifact_name_and_iter_query(self, query, name=None, iter=None):
        if not name and not iter:
            return query

        # Escape special chars (_,%) since we still need to do a like query because of the iter.
        # Also limit length to len(str) + 3, assuming iter is < 100 (two iter digits + hyphen)
        # this helps filter the situations where we match a suffix by mistake due to the like query.
        exact_name = self._escape_characters_for_like_query(name)

        if name and name.startswith("~"):
            # Like query
            iter_prefix = f"{iter}-" if iter else ""
            return query.filter(
                Artifact.key.ilike(f"{iter_prefix}%{exact_name[1:]}%", escape="\\")
            )

        # From here on, it's either exact name match or no name
        if iter:
            if name:
                return query.filter(Artifact.key == f"{iter}-{name}")
            return query.filter(Artifact.key.ilike(f"{iter}-%"))

        # Exact match, no iter specified
        return query.filter(
            or_(
                Artifact.key == name,
                and_(
                    Artifact.key.like(f"%-{exact_name}", escape="\\"),
                    func.length(Artifact.key) < len(name) + 4,
                ),
            )
        )

    def _find_artifacts(
        self,
        session,
        project,
        ids,
        labels=None,
        since=None,
        until=None,
        name=None,
        kind=None,
        category: schemas.ArtifactCategories = None,
        iter=None,
        use_tag_as_uid: bool = None,
    ):
        """
        TODO: refactor this method
        basically ids should be list of strings (representing ids), but we also handle 3 special cases (mainly for
        BC until we refactor the whole artifacts API):
        1. ids == "*" - in which we don't care about ids we just don't add any filter for this column
        2. ids == "latest":
        use_tag_as_uid==(None or False) we find the relevant uid by finding the latest artifact using the updated column
        use_tag_as_uid==True we are treating the ids as uid (for backwards compatibility where we have artifacts which
        were created with uid==latest when created using the project.log_artifact() method)
        3. ids is a string (different than "latest") - in which the meaning is actually a uid, so we add this filter
        """
        if category and kind:
            message = "Category and Kind filters can't be given together"
            logger.warning(message, kind=kind, category=category)
            raise ValueError(message)
        labels = label_set(labels)
        query = self._query(session, Artifact, project=project)
        if ids != "*":
            if ids == "latest" and not use_tag_as_uid:
                query = self._latest_uid_filter(session, query)
            elif isinstance(ids, str):
                query = query.filter(Artifact.uid == ids)
            else:
                query = query.filter(Artifact.id.in_(ids))
        query = self._add_labels_filter(session, query, Artifact, labels)

        if since or until:
            since = since or datetime.min
            until = until or datetime.max
            query = query.filter(
                and_(Artifact.updated >= since, Artifact.updated <= until)
            )

        query = self._add_artifact_name_and_iter_query(query, name, iter)

        if kind:
            return self._filter_artifacts_by_kinds(query, [kind])

        elif category:
            filtered_artifacts = self._filter_artifacts_by_category(query, category)
            # TODO - this is a hack needed since link artifacts will be returned even for artifacts of
            #        the wrong category. Remove this when we refactor this area.
            return self._filter_out_extra_link_artifacts(filtered_artifacts)
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

    # TODO - this is a hack needed since link artifacts will be returned even for artifacts of
    #        the wrong category. Remove this when we refactor this area.
    @staticmethod
    def _filter_out_extra_link_artifacts(artifacts):
        # Only keep link artifacts that point at "real" artifacts that already exist in the results
        existing_keys = set()
        link_artifacts = []
        filtered_artifacts = []
        for artifact in artifacts:
            if artifact.struct.get("kind") != "link":
                existing_keys.add(artifact.key)
                filtered_artifacts.append(artifact)
            else:
                link_artifacts.append(artifact)

        for link_artifact in link_artifacts:
            if "spec" in link_artifact.struct:
                link_iteration = link_artifact.struct["spec"].get("link_iteration")
            else:
                link_iteration = link_artifact.struct.get("link_iteration")
            if not link_iteration:
                continue
            linked_key = f"{link_iteration}-{link_artifact.key}"
            if linked_key in existing_keys:
                filtered_artifacts.append(link_artifact)

        return filtered_artifacts

    def _find_functions(self, session, name, project, uids=None, labels=None):
        query = self._query(session, Function, project=project)
        if name:
            query = query.filter(generate_query_predicate_for_name(Function.name, name))
        if uids is not None:
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
        # Some specific handling is needed for the case of a query like "label=x&label=x=value". In this case
        # of course it should be reduced to "label=x=value". That's why we need to keep the labels that are queried
        # with values, and then remove it from the list of labels queried without value.
        label_names_with_values = set()
        label_names_no_values = set()

        for lbl in labels:
            if "=" in lbl:
                name, value = [v.strip() for v in lbl.split("=", 1)]
                cond = and_(cls.Label.name == name, cls.Label.value == value)
                preds.append(cond)
                label_names_with_values.add(name)
            else:
                label_names_no_values.add(lbl.strip())

        for name in label_names_no_values.difference(label_names_with_values):
            preds.append(cls.Label.name == name)

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
    ) -> schemas.ScheduleRecord:
        schedule = schemas.ScheduleRecord.from_orm(schedule_record)
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
    ) -> schemas.FeatureSet:
        feature_set_full_dict = feature_set_record.full_object
        feature_set_resp = schemas.FeatureSet(**feature_set_full_dict)

        feature_set_resp.metadata.tag = tag
        return feature_set_resp

    @staticmethod
    def _transform_feature_vector_model_to_schema(
        feature_vector_record: FeatureVector,
        tag=None,
    ) -> schemas.FeatureVector:
        feature_vector_full_dict = feature_vector_record.full_object
        feature_vector_resp = schemas.FeatureVector(**feature_vector_full_dict)

        feature_vector_resp.metadata.tag = tag
        feature_vector_resp.metadata.created = feature_vector_record.created
        return feature_vector_resp

    def _transform_project_record_to_schema(
        self, session: Session, project_record: Project
    ) -> schemas.Project:
        # in projects that was created before 0.6.0 the full object wasn't created properly - fix that, and return
        if not project_record.full_object:
            project = schemas.Project(
                metadata=schemas.ProjectMetadata(
                    name=project_record.name,
                    created=project_record.created,
                ),
                spec=schemas.ProjectSpec(
                    description=project_record.description,
                    source=project_record.source,
                ),
                status=schemas.ObjectStatus(
                    state=project_record.state,
                ),
            )
            self.store_project(session, project_record.name, project)
            return project
        # TODO: handle transforming the functions/workflows/artifacts references to real objects
        return schemas.Project(**project_record.full_object)

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

        query = session.query(MarketplaceSource).filter(
            MarketplaceSource.index >= start
        )
        if end:
            query = query.filter(MarketplaceSource.index <= end)

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
    def _transform_marketplace_source_record_to_schema(
        marketplace_source_record: MarketplaceSource,
    ) -> schemas.IndexedMarketplaceSource:
        source_full_dict = marketplace_source_record.full_object
        marketplace_source = schemas.MarketplaceSource(**source_full_dict)
        return schemas.IndexedMarketplaceSource(
            index=marketplace_source_record.index, source=marketplace_source
        )

    @staticmethod
    def _transform_marketplace_source_schema_to_record(
        marketplace_source_schema: schemas.IndexedMarketplaceSource,
        current_object: MarketplaceSource = None,
    ):
        now = datetime.now(timezone.utc)
        if current_object:
            if current_object.name != marketplace_source_schema.source.metadata.name:
                raise mlrun.errors.MLRunInternalServerError(
                    "Attempt to update object while replacing its name"
                )
            created_timestamp = current_object.created
        else:
            created_timestamp = marketplace_source_schema.source.metadata.created or now
        updated_timestamp = marketplace_source_schema.source.metadata.updated or now

        marketplace_source_record = MarketplaceSource(
            id=current_object.id if current_object else None,
            name=marketplace_source_schema.source.metadata.name,
            index=marketplace_source_schema.index,
            created=created_timestamp,
            updated=updated_timestamp,
        )
        full_object = marketplace_source_schema.source.dict()
        full_object["metadata"]["created"] = str(created_timestamp)
        full_object["metadata"]["updated"] = str(updated_timestamp)
        # Make sure we don't keep any credentials in the DB. These are handled in the marketplace crud object.
        full_object["spec"].pop("credentials", None)

        marketplace_source_record.full_object = full_object
        return marketplace_source_record

    @staticmethod
    def _validate_and_adjust_marketplace_order(session, order):
        max_order = session.query(func.max(MarketplaceSource.index)).scalar()
        if not max_order or max_order < 0:
            max_order = 0

        if order == schemas.marketplace.last_source_index:
            order = max_order + 1

        if order > max_order + 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Order must not exceed the current maximal order + 1. max_order = {max_order}, order = {order}"
            )
        if order < 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Order of inserted source must be greater than 0 or "
                + f"{schemas.marketplace.last_source_index} (for last). order = {order}"
            )
        return order

    def create_marketplace_source(
        self, session, ordered_source: schemas.IndexedMarketplaceSource
    ):
        logger.debug(
            "Creating marketplace source in DB",
            index=ordered_source.index,
            name=ordered_source.source.metadata.name,
        )

        order = self._validate_and_adjust_marketplace_order(
            session, ordered_source.index
        )
        name = ordered_source.source.metadata.name
        source_record = self._query(session, MarketplaceSource, name=name).one_or_none()
        if source_record:
            raise mlrun.errors.MLRunConflictError(
                f"Marketplace source name already exists. name = {name}"
            )
        source_record = self._transform_marketplace_source_schema_to_record(
            ordered_source
        )

        self._move_and_reorder_table_items(
            session, source_record, move_to=order, move_from=None
        )

    @retry_on_conflict
    def store_marketplace_source(
        self,
        session,
        name,
        ordered_source: schemas.IndexedMarketplaceSource,
    ):
        logger.debug(
            "Storing marketplace source in DB", index=ordered_source.index, name=name
        )

        if name != ordered_source.source.metadata.name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Conflict between resource name and metadata.name in the stored object"
            )
        order = self._validate_and_adjust_marketplace_order(
            session, ordered_source.index
        )

        source_record = self._query(session, MarketplaceSource, name=name).one_or_none()
        current_order = source_record.index if source_record else None
        if current_order == schemas.marketplace.last_source_index:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Attempting to modify the global marketplace source."
            )
        source_record = self._transform_marketplace_source_schema_to_record(
            ordered_source, source_record
        )

        self._move_and_reorder_table_items(
            session, source_record, move_to=order, move_from=current_order
        )

    def list_marketplace_sources(
        self, session
    ) -> List[schemas.IndexedMarketplaceSource]:
        results = []
        query = self._query(session, MarketplaceSource).order_by(
            MarketplaceSource.index.desc()
        )
        for record in query:
            ordered_source = self._transform_marketplace_source_record_to_schema(record)
            # Need this to make the list return such that the default source is last in the response.
            if ordered_source.index != schemas.last_source_index:
                results.insert(0, ordered_source)
            else:
                results.append(ordered_source)
        return results

    def delete_marketplace_source(self, session, name):
        logger.debug("Deleting marketplace source from DB", name=name)

        source_record = self._query(session, MarketplaceSource, name=name).one_or_none()
        if not source_record:
            return

        current_order = source_record.index
        if current_order == schemas.marketplace.last_source_index:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Attempting to delete the global marketplace source."
            )

        self._move_and_reorder_table_items(
            session, source_record, move_to=None, move_from=current_order
        )

    def get_marketplace_source(self, session, name) -> schemas.IndexedMarketplaceSource:
        source_record = self._query(session, MarketplaceSource, name=name).one_or_none()
        if not source_record:
            raise mlrun.errors.MLRunNotFoundError(
                f"Marketplace source not found. name = {name}"
            )

        return self._transform_marketplace_source_record_to_schema(source_record)

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

    @retry_on_conflict
    def store_background_task(
        self,
        session,
        name: str,
        project: str,
        state: str = mlrun.api.schemas.BackgroundTaskState.running,
        timeout: int = None,
    ):
        background_task_record = self._query(
            session,
            BackgroundTask,
            name=name,
            project=project,
        ).one_or_none()
        now = datetime.now(timezone.utc)
        if background_task_record:
            # we don't want to be able to change state after it reached terminal state
            if (
                background_task_record.state
                in mlrun.api.schemas.BackgroundTaskState.terminal_states()
                and state != background_task_record.state
            ):
                raise mlrun.errors.MLRunRuntimeError(
                    "Background task already reached terminal state, can not change to another state. Failing"
                )

            if timeout and mlrun.mlconf.background_tasks.timeout_mode == "enabled":
                background_task_record.timeout = int(timeout)
            background_task_record.state = state
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
            )
        self._upsert(session, [background_task_record])

    def get_background_task(
        self, session, name: str, project: str
    ) -> schemas.BackgroundTask:
        background_task_record = self._get_background_task_record(
            session, name, project
        )
        if self._is_background_task_timeout_exceeded(background_task_record):
            # lazy update of state, only if get background task was requested and the timeout for the update passed
            # and the task still in progress then we change to failed
            self.store_background_task(
                session,
                name,
                project,
                mlrun.api.schemas.background_task.BackgroundTaskState.failed,
            )
            background_task_record = self._get_background_task_record(
                session, name, project
            )

        return self._transform_background_task_record_to_schema(background_task_record)

    @staticmethod
    def _transform_background_task_record_to_schema(
        background_task_record: BackgroundTask,
    ) -> schemas.BackgroundTask:
        return schemas.BackgroundTask(
            metadata=schemas.BackgroundTaskMetadata(
                name=background_task_record.name,
                project=background_task_record.project,
                created=background_task_record.created,
                updated=background_task_record.updated,
                timeout=background_task_record.timeout,
            ),
            spec=schemas.BackgroundTaskSpec(),
            status=schemas.BackgroundTaskStatus(
                state=background_task_record.state,
            ),
        )

    def _list_project_background_tasks(
        self, session: Session, project: str
    ) -> typing.List[str]:
        return [
            name
            for name, in self._query(
                session, distinct(BackgroundTask.name), project=project
            ).all()
        ]

    def _delete_background_tasks(self, session: Session, project: str):
        logger.debug("Removing background tasks from db", project=project)
        for background_task_name in self._list_project_background_tasks(
            session, project
        ):
            self.delete_background_task(session, background_task_name, project)

    def delete_background_task(self, session: Session, name: str, project: str):
        self._delete(session, BackgroundTask, name=name, project=project)

    def _get_background_task_record(
        self,
        session: Session,
        name: str,
        project: str,
        raise_on_not_found: bool = True,
    ) -> BackgroundTask:
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

    @staticmethod
    def _is_background_task_timeout_exceeded(background_task_record) -> bool:
        # We don't verify if timeout_mode is enabled because if timeout is defined and
        # mlrun.mlconf.background_tasks.timeout_mode == "disabled",
        # it signifies that the background task was initiated while timeout mode was enabled,
        # and we intend to verify it as if timeout mode was enabled
        timeout = background_task_record.timeout
        if (
            timeout
            and background_task_record.state
            not in mlrun.api.schemas.BackgroundTaskState.terminal_states()
            and datetime.utcnow()
            > timedelta(seconds=int(timeout)) + background_task_record.updated
        ):
            return True
        return False
