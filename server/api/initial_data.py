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
import datetime
import json
import os
import pathlib
import typing

import dateutil.parser
import pymysql.err
import sqlalchemy.exc
import sqlalchemy.orm

import mlrun.artifacts
import mlrun.artifacts.base
import mlrun.common.formatters
import mlrun.common.schemas
import server.api.constants
import server.api.crud.pagination_cache
import server.api.db.sqldb.db
import server.api.db.sqldb.helpers
import server.api.db.sqldb.models
import server.api.utils.db.alembic
import server.api.utils.db.backup
import server.api.utils.db.mysql
import server.api.utils.scheduler
from mlrun.artifacts.base import fill_artifact_object_hash
from mlrun.config import config
from mlrun.errors import MLRunPreconditionFailedError, err_to_str
from mlrun.utils import (
    is_legacy_artifact,
    is_link_artifact,
    logger,
)
from server.api.db.init_db import init_db
from server.api.db.session import close_session, create_session
from server.api.db.sqldb.models import ProjectSummary


def init_data(
    from_scratch: bool = False, perform_migrations_if_needed: bool = False
) -> None:
    logger.info("Initializing DB data")

    alembic_util = None

    # create mysql util, and if mlrun is configured to use mysql, wait for it to be live and set its db modes
    mysql_util = server.api.utils.db.mysql.MySQLUtil(logger)
    if mysql_util.get_mysql_dsn_data():
        mysql_util.wait_for_db_liveness()
        mysql_util.set_modes(mlrun.mlconf.httpdb.db.mysql.modes)

        alembic_util = _create_alembic_util()
        (
            is_migration_needed,
            is_migration_from_scratch,
            is_backup_needed,
        ) = _resolve_needed_operations(alembic_util, from_scratch)
    else:
        dsn = mysql_util.get_dsn()
        if "sqlite" in dsn:
            logger.debug("SQLite DB is used, liveness check not needed")
        else:
            logger.warn(
                f"Invalid mysql dsn: {dsn}, assuming live and skipping liveness verification"
            )

        # migration is not needed for sqlite, but we mark it as from scratch to initialize the db
        is_migration_from_scratch = True
        is_migration_needed = False
        is_backup_needed = False

    if (
        not is_migration_from_scratch
        and not perform_migrations_if_needed
        and is_migration_needed
    ):
        state = mlrun.common.schemas.APIStates.waiting_for_migrations
        logger.info("Migration is needed, changing API state", state=state)
        config.httpdb.state = state
        return

    if is_backup_needed:
        logger.info("DB Backup is needed, backing up...")
        db_backup = server.api.utils.db.backup.DBBackupUtil()
        db_backup.backup_database()

    logger.info("Creating initial data")
    config.httpdb.state = mlrun.common.schemas.APIStates.migrations_in_progress

    if is_migration_from_scratch or is_migration_needed:
        try:
            _perform_schema_migrations(alembic_util)
            init_db()
            db_session = create_session()
            try:
                _add_initial_data(db_session)
                _perform_data_migrations(db_session)
            finally:
                close_session(db_session)
        except Exception:
            state = mlrun.common.schemas.APIStates.migrations_failed
            logger.warning("Migrations failed, changing API state", state=state)
            config.httpdb.state = state
            raise
    # if the above process actually ran a migration - initializations that were skipped on the API initialization
    # should happen - we can't do it here because it requires an asyncio loop which can't be accessible here
    # therefore moving to migration_completed state, and other component will take care of moving to online
    if alembic_util and not is_migration_from_scratch and is_migration_needed:
        config.httpdb.state = mlrun.common.schemas.APIStates.migrations_completed
    else:
        config.httpdb.state = mlrun.common.schemas.APIStates.online

    if not from_scratch:
        # Cleanup pagination cache on api startup
        session = create_session()
        server.api.crud.pagination_cache.PaginationCache().cleanup_pagination_cache(
            session
        )
        session.commit()

    logger.info("Initial data created")


# If the data_table version doesn't exist, we can assume the data version is 1.
# This is because data version 1 points to a data migration which was added back in 0.6.0, and
# upgrading from a version earlier than 0.6.0 to v>=0.8.0 is not supported.
data_version_prior_to_table_addition = 1

# NOTE: Bump this number when adding a new data migration
latest_data_version = 7


def update_default_configuration_data():
    server.api.db.session.run_function_with_new_db_session(
        server.api.crud.Alerts().populate_event_cache
    )

    logger.debug("Updating default configuration data")
    db_session = create_session()
    try:
        db = server.api.db.sqldb.db.SQLDB()
        _add_default_hub_source_if_needed(db, db_session)
        _add_default_alert_templates(db, db_session)
    finally:
        close_session(db_session)


def _resolve_needed_operations(
    alembic_util: server.api.utils.db.alembic.AlembicUtil,
    force_from_scratch: bool = False,
) -> tuple[bool, bool, bool]:
    is_migration_from_scratch = (
        force_from_scratch or alembic_util.is_migration_from_scratch()
    )
    is_schema_migration_needed = alembic_util.is_schema_migration_needed()
    is_data_migration_needed = (
        not _is_latest_data_version()
        and config.httpdb.db.data_migrations_mode == "enabled"
    )
    is_migration_needed = not is_migration_from_scratch and (
        is_schema_migration_needed or is_data_migration_needed
    )
    is_backup_needed = (
        config.httpdb.db.backup.mode == "enabled"
        and is_migration_needed
        and not is_migration_from_scratch
    )
    logger.info(
        "Checking if migration is needed",
        is_migration_from_scratch=is_migration_from_scratch,
        is_schema_migration_needed=is_schema_migration_needed,
        is_data_migration_needed=is_data_migration_needed,
        is_backup_needed=is_backup_needed,
        is_migration_needed=is_migration_needed,
    )

    return is_migration_needed, is_migration_from_scratch, is_backup_needed


def _create_alembic_util() -> server.api.utils.db.alembic.AlembicUtil:
    # run schema migrations on existing DB or create it with alembic
    dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    alembic_config_path = dir_path / "alembic.ini"

    alembic_util = server.api.utils.db.alembic.AlembicUtil(
        alembic_config_path, _is_latest_data_version()
    )
    return alembic_util


def _perform_schema_migrations(alembic_util: server.api.utils.db.alembic.AlembicUtil):
    if alembic_util:
        logger.info("Performing schema migration")
        alembic_util.init_alembic()


def _is_latest_data_version():
    db_session = create_session()
    db = server.api.db.sqldb.db.SQLDB()

    try:
        current_data_version = _resolve_current_data_version(db, db_session)
    finally:
        close_session(db_session)

    return current_data_version == latest_data_version


def _perform_data_migrations(db_session: sqlalchemy.orm.Session):
    if config.httpdb.db.data_migrations_mode == "enabled":
        db = server.api.db.sqldb.db.SQLDB()
        current_data_version = int(db.get_current_data_version(db_session))
        if current_data_version != latest_data_version:
            logger.info(
                "Performing data migrations",
                current_data_version=current_data_version,
                latest_data_version=latest_data_version,
            )
            if current_data_version < 1:
                raise MLRunPreconditionFailedError(
                    "Data migration from data version 0 is not supported. "
                    "Upgrade to MLRun <= 1.5.0 before performing this migration"
                )
            if current_data_version < 2:
                _perform_version_2_data_migrations(db, db_session)
            if current_data_version < 3:
                _perform_version_3_data_migrations(db, db_session)
            if current_data_version < 4:
                _perform_version_4_data_migrations(db, db_session)
            if current_data_version < 5:
                _perform_version_5_data_migrations(db, db_session)
            if current_data_version < 6:
                _perform_version_6_data_migrations(db, db_session)
            if current_data_version < 7:
                _perform_version_7_data_migrations(db, db_session)
            if current_data_version < 8:
                _perform_version_8_data_migrations(db, db_session)

            db.create_data_version(db_session, str(latest_data_version))


def _add_initial_data(db_session: sqlalchemy.orm.Session):
    db = server.api.db.sqldb.db.SQLDB()
    _add_data_version(db, db_session)


def _perform_version_2_data_migrations(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _align_runs_table(db, db_session)


def _align_runs_table(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    logger.info("Aligning runs")
    runs = db._find_runs(db_session, None, "*", None).all()
    for run in runs:
        run_dict = run.struct

        # Align run start_time column to the start time from the body
        run.start_time = (
            server.api.db.sqldb.helpers.run_start_time(run_dict) or run.start_time
        )
        # in case no start time was in the body, we took the time from the column, let's make sure the body will have
        # it as well
        run_dict.setdefault("status", {})["start_time"] = (
            db._add_utc_timezone(run.start_time).isoformat() if run.start_time else None
        )

        # New name column added, fill it up from the body
        run.name = run_dict.get("metadata", {}).get("name", "no-name")
        # in case no name was in the body, we defaulted to "no-name", let's make sure the body will have it as well
        run_dict.setdefault("metadata", {})["name"] = run.name

        # State field used to have a bug causing only the body to be updated, align the column
        run.state = run_dict.get("status", {}).get(
            "state", mlrun.common.runtimes.constants.RunStates.created
        )
        # in case no name was in the body, we defaulted to created, let's make sure the body will have it as well
        run_dict.setdefault("status", {})["state"] = run.state

        # New updated column added, fill it up from the body
        updated = datetime.datetime.now(tz=datetime.timezone.utc)
        if run_dict.get("status", {}).get("last_update"):
            updated = dateutil.parser.parse(
                run_dict.get("status", {}).get("last_update")
            )
        db._update_run_updated_time(run, run_dict, updated)
        run.struct = run_dict
        db._upsert(db_session, [run], ignore=True)


def _perform_version_3_data_migrations(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _rename_marketplace_kind_to_hub(db, db_session)


def _rename_marketplace_kind_to_hub(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    logger.info("Renaming 'Marketplace' kinds to 'Hub'")

    hubs = db._list_hub_sources_without_transform(db_session)
    for hub in hubs:
        hub_dict = hub.full_object

        # rename kind from "MarketplaceSource" to "HubSource"
        if "Marketplace" in hub_dict.get("kind", ""):
            hub_dict["kind"] = hub_dict["kind"].replace("Marketplace", "Hub")

        # save the object back to the db
        hub.full_object = hub_dict
        db._upsert(db_session, [hub], ignore=True)


def _perform_version_4_data_migrations(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _update_default_hub_source(db, db_session)


def _add_default_hub_source_if_needed(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    default_hub_source = mlrun.common.schemas.HubSource.generate_default_source()
    # hub_source will be None if the configuration has hub.default_source.create=False
    if not default_hub_source:
        logger.info("Not adding default hub source, per configuration")
        return

    hub_source = db.get_hub_source(
        db_session,
        index=mlrun.common.schemas.hub.last_source_index,
        raise_on_not_found=False,
    )

    # update the default hub if configured url has changed
    hub_source_path = hub_source.source.spec.path if hub_source else None
    if not hub_source_path or hub_source_path != default_hub_source.spec.path:
        logger.debug(
            "Updating default hub source",
            hub_source_path=hub_source_path,
            default_hub_source_path=default_hub_source.spec.path,
        )
        _update_default_hub_source(db, db_session, default_hub_source)


def _update_default_hub_source(
    db: server.api.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    hub_source: mlrun.common.schemas.hub.HubSource = None,
):
    """
    Updates default hub source in db.
    """
    hub_source = hub_source or mlrun.common.schemas.HubSource.generate_default_source()
    if not hub_source:
        logger.info("Not adding default hub source, per configuration")
        return

    _delete_default_hub_source(db_session)
    logger.info("Adding default hub source")
    # Not using db.store_hub_source() since it doesn't allow changing the default hub source.
    hub_record = db._transform_hub_source_schema_to_record(
        mlrun.common.schemas.IndexedHubSource(
            index=mlrun.common.schemas.hub.last_source_index,
            source=hub_source,
        )
    )
    db_session.add(hub_record)
    db_session.commit()


def _delete_default_hub_source(db_session: sqlalchemy.orm.Session):
    """
    Delete default hub source directly from db
    """
    # Not using db.delete_hub_source() since it doesn't allow deleting the default hub source.
    default_record = (
        db_session.query(server.api.db.sqldb.models.HubSource)
        .filter(
            server.api.db.sqldb.models.HubSource.index
            == mlrun.common.schemas.last_source_index
        )
        .one_or_none()
    )
    if default_record:
        logger.info(f"Deleting default hub source {default_record.name}")
        db_session.delete(default_record)
        db_session.commit()
    else:
        logger.info("Default hub source not found")


def _add_data_version(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    if db.get_current_data_version(db_session, raise_on_not_found=False) is None:
        data_version = _resolve_current_data_version(db, db_session)
        logger.info(
            "No data version, setting data version",
            data_version=data_version,
        )
        db.create_data_version(db_session, data_version)


def _resolve_current_data_version(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    try:
        return int(db.get_current_data_version(db_session))
    except (
        sqlalchemy.exc.ProgrammingError,
        sqlalchemy.exc.OperationalError,
        pymysql.err.ProgrammingError,
        pymysql.err.OperationalError,
        mlrun.errors.MLRunNotFoundError,
    ) as exc:
        try:
            projects = db.list_projects(
                db_session, format_=mlrun.common.formatters.ProjectFormat.name_only
            )
        except (
            sqlalchemy.exc.ProgrammingError,
            sqlalchemy.exc.OperationalError,
            pymysql.err.ProgrammingError,
            pymysql.err.OperationalError,
        ):
            projects = None

        # heuristic - if there are no projects it's a new DB - data version is latest
        if not projects or not projects.projects:
            logger.info(
                "No projects in DB, assuming latest data version",
                exc=err_to_str(exc),
                latest_data_version=latest_data_version,
            )
            return latest_data_version
        elif "no such table" in str(exc) or (
            "Table" in str(exc) and "doesn't exist" in str(exc)
        ):
            logger.info(
                "Data version table does not exist, assuming prior version",
                exc=err_to_str(exc),
                data_version_prior_to_table_addition=data_version_prior_to_table_addition,
            )
            return data_version_prior_to_table_addition
        elif isinstance(exc, mlrun.errors.MLRunNotFoundError):
            logger.info(
                "Data version table exist without version, assuming prior version",
                exc=err_to_str(exc),
                data_version_prior_to_table_addition=data_version_prior_to_table_addition,
            )
            return data_version_prior_to_table_addition

        raise exc


def _perform_version_5_data_migrations(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _migrate_artifacts_table_v2(db, db_session)


def _migrate_artifacts_table_v2(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    """
    Migrate the old artifacts table to the new artifacts_v2 table, including their respective tags and labels.
    The migration is done in batches, to not overload the db. A state file is used to keep track of the migration
    progress, and is updated after each batch, so that if the migration fails, it can be resumed from the last batch.
    Delete the old artifacts table when done.
    """

    # count the total number of artifacts to migrate
    total_artifacts_count = db._query(
        db_session, server.api.db.sqldb.models.Artifact
    ).count()

    if total_artifacts_count == 0:
        logger.info("No v1 artifacts in the system, skipping migration")
        return

    logger.info(
        "Migrating artifacts to artifacts_v2 table",
        total_artifacts_count=total_artifacts_count,
    )
    batch_size = config.artifacts.artifact_migration_batch_size

    # get the id of the last migrated artifact and the list of all link artifacts ids from the state file
    last_migrated_artifact_id, link_artifact_ids = _get_migration_state()

    while True:
        # migrate the next batch
        last_migrated_artifact_id, batch_link_artifact_ids = _migrate_artifacts_batch(
            db, db_session, last_migrated_artifact_id, batch_size
        )
        if batch_link_artifact_ids:
            link_artifact_ids.update(batch_link_artifact_ids)

        if last_migrated_artifact_id is None:
            # we're done
            break
        _update_state_file(last_migrated_artifact_id, link_artifact_ids)

    # find the best iteration artifacts the link artifacts point at ,
    # and mark them as best iteration artifacts in the new artifacts_v2 table
    _mark_best_iteration_artifacts(db, db_session, link_artifact_ids)

    # delete the state file
    _delete_state_file()

    logger.debug("Deleting old artifacts table, including their labels and tags")

    # drop the old artifacts table, including their labels and tags tables
    db.delete_table_records(
        db_session, server.api.db.sqldb.models.Artifact.Label, raise_on_not_exists=False
    )
    db.delete_table_records(
        db_session, server.api.db.sqldb.models.Artifact.Tag, raise_on_not_exists=False
    )
    db.delete_table_records(
        db_session, server.api.db.sqldb.models.Artifact, raise_on_not_exists=False
    )

    logger.info("Finished migrating artifacts to artifacts_v2 table successfully")


def _migrate_artifacts_batch(
    db: server.api.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    last_migrated_artifact_id: int,
    batch_size: int,
):
    new_artifacts = []
    old_id_to_artifact = {}
    artifacts_labels_to_migrate = []
    link_artifact_ids = []

    # get artifacts from the db, sorted by id
    query = db._query(db_session, server.api.db.sqldb.models.Artifact)
    if last_migrated_artifact_id > 0:
        # skip the artifacts that were already migrated
        query = query.filter(
            server.api.db.sqldb.models.Artifact.id > last_migrated_artifact_id
        )

    query = query.order_by(server.api.db.sqldb.models.Artifact.id).limit(batch_size)

    artifacts = query.all()

    if len(artifacts) == 0:
        # we're done
        return None, None

    logger.debug("Migrating artifacts batch", batch_size=len(artifacts))

    for artifact in artifacts:
        new_artifact = server.api.db.sqldb.models.ArtifactV2()

        artifact_dict = artifact.struct

        if is_legacy_artifact(artifact_dict):
            # convert the legacy artifact to the new format, by setting a metadata field and spec field
            # and copying the old fields to the spec
            artifact_dict = mlrun.artifacts.base.convert_legacy_artifact_to_new_format(
                artifact_dict
            ).to_dict()

        # if it is a link artifact, keep its id. we will use it later to update the best iteration artifacts
        if is_link_artifact(artifact_dict):
            link_artifact_ids.append(artifact.id)
            continue

        artifact_metadata = artifact_dict.get("metadata", None) or {}

        # producer_id - the current uid value
        # uid can be in the metadata or in the artifact itself, or in the tree field
        old_uid = artifact_metadata.get("uid", None)
        if not old_uid:
            old_uid = artifact_dict.get("uid", None)
        if not old_uid:
            old_uid = artifact_metadata.get("tree", None)
        new_artifact.producer_id = old_uid

        # project - copy as is
        new_artifact.project = artifact_metadata.get("project", None)

        # iteration - the artifact's iteration
        iteration = artifact_metadata.get("iter", None)
        new_artifact.iteration = int(iteration) if iteration else 0

        # key - retain the db key to ensure BC of reading artifacts by the index key.
        # if iteration is concatenated to the key, remove it as this was only handled in the backend,
        # and now the iteration is saved in a separate column
        key = artifact.key
        if iteration and key.startswith(f"{iteration}-"):
            key = key[len(f"{iteration}-") :]
        new_artifact.key = key

        # best iteration
        # if iteration == 0 it means it is from a single run since link artifacts were already
        # handled above - so we can set is as best iteration.
        # otherwise set to false, the best iteration artifact will be updated later
        if new_artifact.iteration == 0:
            new_artifact.best_iteration = True
        else:
            new_artifact.best_iteration = False

        # to overcome issues with legacy artifacts with missing keys, we will set the key in the metadata
        if not artifact_metadata.get("key"):
            artifact_dict.setdefault("metadata", {})
            artifact_dict["metadata"]["key"] = key

        # uid - calculate as the hash of the artifact object
        uid = fill_artifact_object_hash(
            artifact_dict, new_artifact.iteration, new_artifact.producer_id
        )
        new_artifact.uid = uid

        # kind - doesn't exist in v1, will be set to "artifact" by default
        new_artifact.kind = artifact_dict.get("kind", mlrun.artifacts.Artifact.kind)

        # updated - the artifact's updated time
        updated = artifact_metadata.get("updated", datetime.datetime.now())
        new_artifact.updated = updated

        # created - the artifact's created time
        # since this is a new field, we just take the updated time
        new_artifact.created = updated

        # full_object - the artifact dict
        new_artifact.full_object = artifact_dict

        # save the new object to the db
        new_artifacts.append(new_artifact)

        last_migrated_artifact_id = artifact.id

        # keep the old tag to artifact mapping, so we can migrate the tags later
        old_id_to_artifact[artifact.id] = new_artifact

        # save the artifact's labels to migrate them later
        labels = artifact_metadata.get("labels", {})
        if labels:
            artifacts_labels_to_migrate.append((new_artifact, labels))

    # add the new artifacts to the db session
    db_session.add_all(new_artifacts)

    # commit the new artifacts first, so they will get an id that can be used when creating tags and labels
    db._commit(db_session, new_artifacts)

    # migrate artifact labels to the new table ("artifact_v2_labels")
    new_labels = _migrate_artifact_labels(db_session, artifacts_labels_to_migrate)

    # migrate artifact tags to the new table ("artifact_v2_tags")
    new_tags = _migrate_artifact_tags(db_session, old_id_to_artifact)

    # commit the new labels and tags
    db._commit(db_session, new_labels + new_tags)

    return last_migrated_artifact_id, link_artifact_ids


def _migrate_artifact_labels(
    db_session: sqlalchemy.orm.Session,
    artifacts_labels_to_migrate: list,
):
    if not artifacts_labels_to_migrate:
        return []

    labels = []

    # iterate over all the artifacts, and create labels for each one
    for artifact, artifacts_labels in artifacts_labels_to_migrate:
        for name, value in artifacts_labels.items():
            new_label = artifact.Label(
                name=name,
                value=value,
                parent=artifact.id,
            )
            labels.append(new_label)
    if labels:
        db_session.add_all(labels)
    return labels


def _migrate_artifact_tags(
    db_session: sqlalchemy.orm.Session,
    old_id_to_artifact: dict[typing.Any, server.api.db.sqldb.models.ArtifactV2],
):
    if not old_id_to_artifact:
        return []

    new_tags = []

    # get all tags that are attached to the artifacts we migrated
    old_tags = (
        db_session.query(server.api.db.sqldb.models.Artifact.Tag)
        .filter(
            server.api.db.sqldb.models.Artifact.Tag.obj_id.in_(
                old_id_to_artifact.keys()
            )
        )
        .all()
    )

    for old_tag in old_tags:
        new_artifact = old_id_to_artifact[old_tag.obj_id]

        # create a new tag object
        new_tag = server.api.db.sqldb.models.ArtifactV2.Tag(
            project=new_artifact.project,
            name=old_tag.name,
            obj_name=new_artifact.key,
            obj_id=new_artifact.id,
        )
        new_tags.append(new_tag)

    if new_tags:
        db_session.add_all(new_tags)

    return new_tags


def _mark_best_iteration_artifacts(
    db: server.api.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    link_artifact_ids: list,
):
    artifacts_to_commit = []

    # get all link artifacts
    link_artifacts = (
        db_session.query(server.api.db.sqldb.models.Artifact)
        .filter(server.api.db.sqldb.models.Artifact.id.in_(link_artifact_ids))
        .all()
    )

    # get all the artifacts that are attached to the link artifacts
    for link_artifact in link_artifacts:
        link_artifact_dict = link_artifact.struct
        if is_legacy_artifact(link_artifact_dict):
            # convert the legacy artifact to the new format, so we can use the same logic
            link_artifact_dict = (
                mlrun.artifacts.base.convert_legacy_artifact_to_new_format(
                    link_artifact_dict
                ).to_dict()
            )

        # get the artifacts attached to the link artifact
        # if the link key was set explicitly, we should use it to find the artifacts, otherwise use the artifact's key
        link_artifact_key = link_artifact_dict.get("spec").get(
            "link_key", None
        ) or link_artifact_dict.get("key", None)
        link_iteration = link_artifact_dict.get("spec").get("link_iteration", None)
        link_tree = link_artifact_dict.get("spec").get("link_tree", None)

        if not link_iteration:
            logger.warning(
                "Link artifact is missing link iteration, skipping",
                link_artifact_key=link_artifact_key,
                link_artifact_id=link_artifact.id,
            )
            continue

        # get the artifacts attached to the link artifact
        query = db._query(db_session, server.api.db.sqldb.models.ArtifactV2).filter(
            server.api.db.sqldb.models.ArtifactV2.key == link_artifact_key,
            server.api.db.sqldb.models.ArtifactV2.iteration == link_iteration,
        )
        if link_tree:
            query = query.filter(
                server.api.db.sqldb.models.ArtifactV2.producer_id == link_tree
            )

        artifact = query.one_or_none()
        if not artifact:
            logger.warning(
                "Link artifact is pointing to a non-existent artifact, skipping",
                link_artifact_key=link_artifact_key,
                link_iteration=link_iteration,
                link_artifact_id=link_artifact.id,
            )
            continue

        artifact.best_iteration = True
        artifacts_to_commit.append(artifact)

    db._commit(db_session, artifacts_to_commit)


def _get_migration_state():
    """
    Get the id of the last migrated artifact from the state file.
    If the state file does not exist, return 0.
    """
    try:
        with open(config.artifacts.artifact_migration_state_file_path) as state_file:
            state = json.load(state_file)
            return state.get("last_migrated_id", 0), set(
                state.get("link_artifact_ids", [])
            )
    except FileNotFoundError:
        return 0, set()


def _update_state_file(last_migrated_id: int, link_artifact_ids: set):
    """Create or update the state file with the given batch index.

    :param last_migrated_id: The id of the last migrated artifact.
    """
    state_file_path = config.artifacts.artifact_migration_state_file_path
    state_file_dir = os.path.dirname(state_file_path)
    if not os.path.exists(state_file_dir):
        os.makedirs(state_file_dir)
    with open(state_file_path, "w") as state_file:
        state = {
            "last_migrated_id": last_migrated_id,
            "link_artifact_ids": list(link_artifact_ids),
        }
        json.dump(state, state_file)


def _delete_state_file():
    """Delete the state file."""
    try:
        os.remove(config.artifacts.artifact_migration_state_file_path)
    except FileNotFoundError:
        pass


def _add_default_alert_templates(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    for template in server.api.constants.pre_defined_templates:
        record = db.get_alert_template(db_session, template.template_name)
        if record is None or record.templates_differ(template):
            db.store_alert_template(db_session, template)


def _perform_version_6_data_migrations(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _migrate_model_monitoring_jobs(db, db_session)


def _migrate_model_monitoring_jobs(db, db_session):
    db.delete_schedules(
        session=db_session,
        project="*",
        names=["model-monitoring-controller", "model-monitoring-batch"],
    )
    db.delete_functions(
        session=db_session,
        project="*",
        names=["model-monitoring-controller", "model-monitoring-batch"],
    )


def _perform_version_7_data_migrations(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _create_project_summaries(db, db_session)


def _perform_version_8_data_migrations(
    db: server.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    db.align_schedule_labels(session=db_session)


def _create_project_summaries(db, db_session):
    # Create a project summary record for all projects.
    # We need to create them manually because a summary record is created only when a new
    # project is created, so project that existing prior to the upgrade don't have summaries.
    projects = db.list_projects(
        db_session, format_=mlrun.common.formatters.ProjectFormat.name_only
    )
    project_summaries = [
        ProjectSummary(
            project=project_name,
            summary=mlrun.common.schemas.ProjectSummary(name=project_name).dict(),
        )
        for project_name in projects.projects
    ]
    db._upsert(db_session, project_summaries, ignore=True)


def main() -> None:
    init_data()


if __name__ == "__main__":
    main()
