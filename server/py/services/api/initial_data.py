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

import datetime
import json
import os
import pathlib
import random
import string
import time
import typing
import uuid

import alembic
import alembic.command
import alembic.config
import pymysql.err
import sqlalchemy.engine
import sqlalchemy.exc
import sqlalchemy.orm
import sqlalchemy_utils
from sqlalchemy import and_, exists, not_

import mlrun.artifacts
import mlrun.artifacts.base
import mlrun.common.db.dialects
import mlrun.common.formatters
import mlrun.common.runtimes
import mlrun.common.schemas
import mlrun.config
import mlrun.errors
import mlrun.utils
import mlrun.utils.regex

import framework.constants
import framework.db
import framework.db.session
import framework.db.sqldb.db
import framework.db.sqldb.helpers
import framework.db.sqldb.models
import framework.db.sqldb.sql_session
import framework.utils.pagination_cache
import services.api.utils.db.alembic
import services.api.utils.db.backup
import services.api.utils.events.events_factory
import services.api.utils.scheduler
from framework.utils.db.utils import DBUtil


def init_data(
    perform_migrations_if_needed: bool = False,
) -> None:
    """
    This function initializes the database with the necessary data.
    It checks if the database exists and if it has any tables,
    and if not, it creates the database and initializes it from scratch.
    """
    mlrun.utils.logger.info("Initializing DB data")

    engine = framework.db.sqldb.sql_session.get_engine()
    db_initialized = _initialize_db_if_needed(engine)

    if db_initialized:
        mlrun.utils.logger.info("Creating database from scratch")
        _initialize_db_from_scratch(engine)
    else:
        mlrun.utils.logger.info("Migrating existing data")
        _migrate_existing_data(
            engine,
            perform_migrations_if_needed,
        )

    mlrun.utils.logger.info("Initial data created")


def _initialize_db_if_needed(engine: sqlalchemy.engine.Engine) -> bool:
    """
    Checks if the database instance exists and is initialized.
    Returns True if the database needs to be created or initialized from scratch (i.e.,
    if the database does not exist or exists but has no tables).
    Returns False if the database exists and has tables (i.e., is set up and ready).
    """
    url = engine.url
    if not sqlalchemy_utils.database_exists(url):
        mlrun.utils.logger.info(
            "Database does not exist, creating",
            database_url=url,
        )
        sqlalchemy_utils.create_database(url)
        return True

    # db exists, lets see if it has some tables
    has_tables = bool(sqlalchemy.inspect(engine).get_table_names())
    if not has_tables:
        mlrun.utils.logger.info(
            "No tables found in the database",
            database_url=url,
        )
        return True

    # db exists and have tables. nothing to ensure
    return False


def _create_schema(
    engine: sqlalchemy.engine.Engine | None = None,
) -> None:
    if engine is None:
        engine = framework.db.sqldb.sql_session.get_engine()

    framework.db.sqldb.models.Base.metadata.create_all(bind=engine)


def _initialize_db_from_scratch(
    engine: sqlalchemy.engine.Engine,
):
    url = engine.url
    _create_schema(
        engine=engine,
    )
    cfg = alembic.config.Config(str(pathlib.Path(__file__).parent / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", str(url))
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        alembic.command.stamp(cfg, "head")

    db = framework.db.sqldb.db.SQLDB()
    framework.db.session.run_function_with_new_db_session(
        func=db.create_data_version,
        version=str(latest_data_version),
    )
    mlrun.mlconf.httpdb.state = mlrun.common.schemas.APIStates.online
    # initialize system id
    framework.db.session.run_function_with_new_db_session(func=_init_system_id)


def _migrate_existing_data(
    engine: sqlalchemy.engine.Engine,
    perform_migrations_if_needed: bool = False,
):
    alembic_util = _create_alembic_util()
    db_util = DBUtil()
    db_util.wait_for_db_liveness()
    db_util.set_configurations()
    migration_operations: dict[str, tuple] = {}
    if engine.name != mlrun.common.db.dialects.Dialects.SQLITE:
        (
            is_migration_needed,
            is_backup_needed,
            migration_operations,
        ) = _resolve_needed_operations(alembic_util)
    else:
        # ON SQLite, we don't have schema migrations, so we don't need to check for them
        is_migration_needed = False
        is_backup_needed = False

    migration_scope, migration_versions = _operations_to_scope_and_versions(
        migration_operations
    )

    if not perform_migrations_if_needed and is_migration_needed:
        state = mlrun.common.schemas.APIStates.waiting_for_migrations
        mlrun.utils.logger.info("Migration is needed, changing API state", state=state)
        mlrun.mlconf.httpdb.state = state
        _publish_db_migration_event(
            mlrun.common.schemas.MigrationEventActions.required,
            scope=migration_scope,
            versions=migration_versions,
        )
        return

    mlrun.utils.logger.info("Creating initial data")
    mlrun.mlconf.httpdb.state = mlrun.common.schemas.APIStates.migrations_in_progress

    db_session = framework.db.session.create_session()
    try:
        if is_migration_needed:
            migration_start_monotonic = time.monotonic()
            _publish_db_migration_event(
                mlrun.common.schemas.MigrationEventActions.started,
                scope=migration_scope,
                versions=migration_versions,
            )
            try:
                # DB backup runs inside the migration try/except so a backup
                # failure transitions the API to migrations_failed and emits
                # the Failed event — it's part of the migration window from
                # an operator's perspective.
                if is_backup_needed:
                    mlrun.utils.logger.info("DB Backup is needed, backing up...")
                    db_backup = services.api.utils.db.backup.DBBackupUtil()
                    db_backup.backup_database()
                _perform_schema_migrations(alembic_util)
                _add_initial_data(db_session)
                _perform_data_migrations(db_session)
            except Exception as exc:
                state = mlrun.common.schemas.APIStates.migrations_failed
                mlrun.utils.logger.warning(
                    "Migrations failed, changing API state", state=state
                )
                mlrun.mlconf.httpdb.state = state
                _publish_db_migration_event(
                    mlrun.common.schemas.MigrationEventActions.failed,
                    error=exc,
                    duration_seconds=_elapsed_since(migration_start_monotonic),
                    scope=migration_scope,
                    versions=migration_versions,
                )
                raise
            _publish_db_migration_event(
                mlrun.common.schemas.MigrationEventActions.completed,
                duration_seconds=_elapsed_since(migration_start_monotonic),
                scope=migration_scope,
                versions=migration_versions,
            )
    finally:
        framework.db.session.close_session(db_session)

    # if the above process actually ran a migration - initializations that were skipped on the API initialization
    # should happen - we can't do it here because it requires an asyncio loop which can't be accessible here
    # therefore moving to migration_completed state, and other component will take care of moving to online
    if alembic_util and is_migration_needed:
        mlrun.mlconf.httpdb.state = mlrun.common.schemas.APIStates.migrations_completed
    else:
        mlrun.mlconf.httpdb.state = mlrun.common.schemas.APIStates.online

    # Cleanup pagination cache on api startup
    framework.db.session.run_function_with_new_db_session(
        func=framework.utils.pagination_cache.PaginationCache().cleanup_pagination_cache,
    )


# If the data_table version doesn't exist, we can assume the data version is 1.
# This is because data version 1 points to a data migration which was added back in 0.6.0, and
# upgrading from a version earlier than 0.6.0 to v>=0.8.0 is not supported.
data_version_prior_to_table_addition = 1

# NOTE: Bump this number when adding a new data migration
latest_data_version = 10


def update_default_configuration_data():
    mlrun.utils.logger.debug("Updating default configuration data")
    db_session = framework.db.session.create_session()
    try:
        db = framework.db.sqldb.db.SQLDB()
        _add_default_hub_source_if_needed(db, db_session)
    finally:
        framework.db.session.close_session(db_session)


def _resolve_needed_operations(
    alembic_util: services.api.utils.db.alembic.AlembicUtil,
) -> tuple[bool, bool, dict[str, tuple]]:
    is_schema_migration_needed = alembic_util.is_schema_migration_needed()
    is_data_migration_needed = (
        not _is_latest_data_version()
        and mlrun.mlconf.httpdb.db.data_migrations_mode == "enabled"
    )
    is_migration_needed = is_schema_migration_needed or is_data_migration_needed
    is_backup_needed = (
        mlrun.mlconf.httpdb.db.backup.mode == "enabled" and is_migration_needed
    )

    operations: dict[str, tuple] = {}
    if is_schema_migration_needed:
        operations["schema"] = (
            alembic_util.get_current_revision(),
            alembic_util.latest_revision,
        )
    if is_data_migration_needed:
        operations["data"] = (
            _get_current_data_version(),
            latest_data_version,
        )

    mlrun.utils.logger.info(
        "Checking if migration is needed",
        is_schema_migration_needed=is_schema_migration_needed,
        is_data_migration_needed=is_data_migration_needed,
        is_backup_needed=is_backup_needed,
        is_migration_needed=is_migration_needed,
    )

    return is_migration_needed, is_backup_needed, operations


def _get_current_data_version() -> int | None:
    db_session = framework.db.session.create_session()
    db = framework.db.sqldb.db.SQLDB()
    try:
        version = _resolve_current_data_version(db, db_session)
    finally:
        framework.db.session.close_session(db_session)
    if version is None:
        return None
    try:
        return int(version)
    except (TypeError, ValueError) as exc:
        mlrun.utils.logger.warning(
            "Could not parse current data version, treating as unknown",
            version=version,
            exc=mlrun.errors.err_to_str(exc),
        )
        return None


def _create_alembic_util() -> services.api.utils.db.alembic.AlembicUtil:
    # run schema migrations on existing DB or create it with alembic
    dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    alembic_config_path = dir_path / "alembic.ini"

    alembic_util = services.api.utils.db.alembic.AlembicUtil(
        alembic_config_path, _is_latest_data_version()
    )
    return alembic_util


def _perform_schema_migrations(alembic_util: services.api.utils.db.alembic.AlembicUtil):
    if alembic_util:
        mlrun.utils.logger.info("Performing schema migration")
        alembic_util.init_alembic()


def _publish_db_migration_event(
    action: mlrun.common.schemas.MigrationEventActions,
    error: BaseException | None = None,
    duration_seconds: float | None = None,
    scope: list[str] | None = None,
    versions: dict | None = None,
) -> None:
    """
    Best-effort publish of a DB migration lifecycle event.
    Failures must not break startup or the migration flow.
    """
    try:
        events_client = (
            services.api.utils.events.events_factory.EventsFactory.get_events_client()
        )
        event = events_client.generate_db_migration_event(
            action,
            error=error,
            duration_seconds=duration_seconds,
            scope=scope,
            versions=versions,
        )
        if event is None:
            return
        events_client.emit(event)
    except Exception as publish_exc:
        mlrun.utils.logger.warning(
            "Failed to publish DB migration event",
            action=action,
            exc=mlrun.errors.err_to_str(publish_exc),
        )


def _elapsed_since(start_monotonic: float | None) -> float | None:
    if start_monotonic is None:
        return None
    return time.monotonic() - start_monotonic


def _operations_to_scope_and_versions(
    operations: dict[str, tuple],
) -> tuple[list[str], dict]:
    versions: dict = {}
    if "schema" in operations:
        current, target = operations["schema"]
        versions["current_schema_revision"] = current
        versions["target_schema_revision"] = target
    if "data" in operations:
        current, target = operations["data"]
        versions["current_data_version"] = current
        versions["target_data_version"] = target
    return list(operations.keys()), versions


def _is_latest_data_version():
    db_session = framework.db.session.create_session()
    db = framework.db.sqldb.db.SQLDB()

    try:
        current_data_version = _resolve_current_data_version(db, db_session)
    finally:
        framework.db.session.close_session(db_session)

    return current_data_version == latest_data_version


def _perform_data_migrations(db_session: sqlalchemy.orm.Session):
    if mlrun.mlconf.httpdb.db.data_migrations_mode == "enabled":
        db = framework.db.sqldb.db.SQLDB()
        current_data_version = int(db.get_current_data_version(db_session))
        if current_data_version != latest_data_version:
            mlrun.utils.logger.info(
                "Performing data migrations",
                current_data_version=current_data_version,
                latest_data_version=latest_data_version,
            )
            if current_data_version < 1:
                raise mlrun.errors.MLRunPreconditionFailedError(
                    "Data migration from data version 0 is not supported. "
                    "Upgrade to MLRun <= 1.5.0 before performing this migration"
                )
            if current_data_version < 5:
                raise mlrun.errors.MLRunPreconditionFailedError(
                    "Data migration from data version less than 5 is not supported. "
                    "Upgrade to MLRun < 1.10.0 before performing this migration"
                )
            if current_data_version < 6:
                _perform_version_6_data_migrations(db, db_session)
            if current_data_version < 7:
                _perform_version_7_data_migrations(db, db_session)
            if current_data_version < 8:
                _perform_version_8_data_migrations(db, db_session)
            if current_data_version < 9:
                _perform_version_9_data_migrations(db, db_session)
            if current_data_version < 10:
                _perform_version_10_data_migrations(db, db_session)

            db.create_data_version(db_session, str(latest_data_version))


def _add_initial_data(db_session: sqlalchemy.orm.Session):
    db = framework.db.sqldb.db.SQLDB()
    _add_data_version(db, db_session)
    # initialize system id
    framework.db.session.run_function_with_new_db_session(func=_init_system_id)


def _add_default_hub_source_if_needed(
    db: framework.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    default_hub_source = mlrun.common.schemas.HubSource.generate_default_source()
    # hub_source will be None if the configuration has hub.default_source.create=False
    if not default_hub_source:
        mlrun.utils.logger.info("Not adding default hub source, per configuration")
        return

    hub_source = db.get_hub_source(
        db_session,
        index=mlrun.common.schemas.hub.last_source_index,
        raise_on_not_found=False,
    )
    if not hub_source:
        # create the default hub source if it does not exist
        _update_default_hub_source(db, db_session, default_hub_source)
        return
    # update the default hub if configuration has changed
    if difference := default_hub_source.diff(hub_source.source):
        mlrun.utils.logger.debug(
            "Updating default hub source",
            difference=difference,
        )
        _update_default_hub_source(db, db_session, default_hub_source)


def _update_default_hub_source(
    db: framework.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    hub_source: mlrun.common.schemas.hub.HubSource = None,
):
    """
    Updates default hub source in db.
    """
    hub_source = hub_source or mlrun.common.schemas.HubSource.generate_default_source()
    if not hub_source:
        mlrun.utils.logger.info("Not adding default hub source, per configuration")
        return

    _delete_default_hub_source(db_session)
    mlrun.utils.logger.info("Adding default hub source")
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
        db_session.query(framework.db.sqldb.models.HubSource)
        .filter(
            framework.db.sqldb.models.HubSource.index
            == mlrun.common.schemas.last_source_index
        )
        .one_or_none()
    )
    if default_record:
        mlrun.utils.logger.info(f"Deleting default hub source {default_record.name}")
        db_session.delete(default_record)
        db_session.commit()
    else:
        mlrun.utils.logger.info("Default hub source not found")


def _add_data_version(
    db: framework.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    if db.get_current_data_version(db_session, raise_on_not_found=False) is None:
        data_version = _resolve_current_data_version(db, db_session)
        mlrun.utils.logger.info(
            "No data version, setting data version",
            data_version=data_version,
        )
        db.create_data_version(db_session, data_version)


def _resolve_current_data_version(
    db: framework.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
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
            mlrun.utils.logger.info(
                "No projects in DB, assuming latest data version",
                exc=mlrun.errors.err_to_str(exc),
                latest_data_version=latest_data_version,
            )
            return latest_data_version
        elif "no such table" in str(exc) or (
            "Table" in str(exc) and "doesn't exist" in str(exc)
        ):
            mlrun.utils.logger.info(
                "Data version table does not exist, assuming prior version",
                exc=mlrun.errors.err_to_str(exc),
                data_version_prior_to_table_addition=data_version_prior_to_table_addition,
            )
            return data_version_prior_to_table_addition
        elif isinstance(exc, mlrun.errors.MLRunNotFoundError):
            mlrun.utils.logger.info(
                "Data version table exist without version, assuming prior version",
                exc=mlrun.errors.err_to_str(exc),
                data_version_prior_to_table_addition=data_version_prior_to_table_addition,
            )
            return data_version_prior_to_table_addition

        raise exc


def _migrate_artifacts_batch(
    db: framework.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    last_migrated_artifact_id: int,
    batch_size: int,
):
    new_artifacts = []
    old_id_to_artifact = {}
    artifacts_labels_to_migrate = []
    link_artifact_ids = []

    # get artifacts from the db, sorted by id
    query = db._query(db_session, framework.db.sqldb.models.Artifact)
    if last_migrated_artifact_id > 0:
        # skip the artifacts that were already migrated
        query = query.filter(
            framework.db.sqldb.models.Artifact.id > last_migrated_artifact_id
        )

    query = query.order_by(framework.db.sqldb.models.Artifact.id).limit(batch_size)

    artifacts = query.all()

    if len(artifacts) == 0:
        # we're done
        return None, None

    mlrun.utils.logger.debug("Migrating artifacts batch", batch_size=len(artifacts))

    for artifact in artifacts:
        new_artifact = framework.db.sqldb.models.ArtifactV2()

        artifact_dict = artifact.struct

        # if it is a link artifact, keep its id. we will use it later to update the best iteration artifacts
        if mlrun.utils.is_link_artifact(artifact_dict):
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
        uid = mlrun.artifacts.base.fill_artifact_object_hash(
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


def _migrate_monitoring_functions_labels(
    db: framework.db.sqldb.db.SQLDB, db_session, chunk_size: int = 500
):
    """
    Update labels for model monitoring infra functions.
    """
    mm_infra_function_names = (
        mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.list()
    )

    def filter_infra_func():
        # filter model monitoring functions which doesn't have the label yet
        functions = framework.db.sqldb.models.Function
        functions_labels = functions.Label  # dynamically generated by the LabelMixin
        return and_(
            functions.name.in_(mm_infra_function_names),
            not_(
                exists().where(
                    and_(
                        functions_labels.parent == functions.id,
                        functions_labels.name
                        == mlrun.common.schemas.ModelMonitoringInfraLabel.KEY,
                    )
                )
            ),
        )

    def add_infra_label(record):
        function_dict = record.struct
        function_metadata_labels_dict = function_dict.get("metadata", {}).get(
            "labels", {}
        )
        # Add a new label to the function metadata
        function_metadata_labels_dict[
            mlrun.common.schemas.ModelMonitoringInfraLabel.KEY
        ] = mlrun.common.schemas.ModelMonitoringInfraLabel.VAL
        function_dict["metadata"]["labels"] = function_metadata_labels_dict

        record.struct = function_dict

        # Generate a new label object
        new_label = framework.db.sqldb.models.Function.Label(
            name=mlrun.common.schemas.ModelMonitoringInfraLabel.KEY,
            value=mlrun.common.schemas.ModelMonitoringInfraLabel.VAL,
            parent=record.id,
        )

        return record, new_label

    return _migrate_data(
        db=db,
        db_session=db_session,
        model=framework.db.sqldb.models.Function,
        filter_func=filter_infra_func,
        handle_field_record_func=add_infra_label,
        chunk_size=chunk_size,
    )


def _migrate_artifact_tags(
    db_session: sqlalchemy.orm.Session,
    old_id_to_artifact: dict[typing.Any, framework.db.sqldb.models.ArtifactV2],
):
    if not old_id_to_artifact:
        return []

    new_tags = []

    # get all tags that are attached to the artifacts we migrated
    old_tags = (
        db_session.query(framework.db.sqldb.models.Artifact.Tag)
        .filter(
            framework.db.sqldb.models.Artifact.Tag.obj_id.in_(old_id_to_artifact.keys())
        )
        .all()
    )

    for old_tag in old_tags:
        new_artifact = old_id_to_artifact[old_tag.obj_id]

        # create a new tag object
        new_tag = framework.db.sqldb.models.ArtifactV2.Tag(
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
    db: framework.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    link_artifact_ids: list,
):
    artifacts_to_commit = []

    # get all link artifacts
    link_artifacts = (
        db_session.query(framework.db.sqldb.models.Artifact)
        .filter(framework.db.sqldb.models.Artifact.id.in_(link_artifact_ids))
        .all()
    )

    # get all the artifacts that are attached to the link artifacts
    for link_artifact in link_artifacts:
        link_artifact_dict = link_artifact.struct

        # get the artifacts attached to the link artifact
        # if the link key was set explicitly, we should use it to find the artifacts, otherwise use the artifact's key
        link_artifact_key = link_artifact_dict.get("spec").get(
            "link_key", None
        ) or link_artifact_dict.get("key", None)
        link_iteration = link_artifact_dict.get("spec").get("link_iteration", None)
        link_tree = link_artifact_dict.get("spec").get("link_tree", None)

        if not link_iteration:
            mlrun.utils.logger.warning(
                "Link artifact is missing link iteration, skipping",
                link_artifact_key=link_artifact_key,
                link_artifact_id=link_artifact.id,
            )
            continue

        # get the artifacts attached to the link artifact
        query = db._query(db_session, framework.db.sqldb.models.ArtifactV2).filter(
            framework.db.sqldb.models.ArtifactV2.key == link_artifact_key,
            framework.db.sqldb.models.ArtifactV2.iteration == link_iteration,
        )
        if link_tree:
            query = query.filter(
                framework.db.sqldb.models.ArtifactV2.producer_id == link_tree
            )

        artifact = query.one_or_none()
        if not artifact:
            mlrun.utils.logger.warning(
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
        with open(
            mlrun.mlconf.artifacts.artifact_migration_state_file_path
        ) as state_file:
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
    state_file_path = mlrun.mlconf.artifacts.artifact_migration_state_file_path
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
        os.remove(mlrun.mlconf.artifacts.artifact_migration_state_file_path)
    except FileNotFoundError:
        pass


def _perform_version_6_data_migrations(
    db: framework.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
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
    db: framework.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _create_project_summaries(db, db_session)


def _perform_version_8_data_migrations(
    db: framework.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    db.align_schedule_labels(session=db_session)


def _perform_version_9_data_migrations(
    db: framework.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _ensure_function_kind_and_state(db, db_session)
    _add_producer_uri_to_artifact(db, db_session)
    _ensure_latest_tag_for_artifacts(db_session)


def _perform_version_10_data_migrations(
    db: framework.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _migrate_monitoring_functions_labels(db, db_session)


def _ensure_function_kind_and_state(
    db: framework.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    chunk_size: int = 500,
):
    def handle_function_kind_and_state(record):
        function_dict = record.struct
        # Since we filter by no kind or no state, check which attribute is the one missing or both are
        if not record.kind:
            record.kind = function_dict.pop("kind", "")
        if not record.state:
            record.state = function_dict.get("status", {}).pop("state", "")
        record.struct = function_dict
        return record

    def filter_function_kind_or_state():
        return getattr(framework.db.sqldb.models.Function, "kind").is_(None) | getattr(
            framework.db.sqldb.models.Function, "state"
        ).is_(None)

    _migrate_data(
        db,
        db_session,
        model=framework.db.sqldb.models.Function,
        filter_func=filter_function_kind_or_state,
        handle_field_record_func=handle_function_kind_and_state,
        chunk_size=chunk_size,
    )


def _add_producer_uri_to_artifact(
    db: framework.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    chunk_size: int | None = None,
):
    chunk_size = chunk_size or mlrun.mlconf.artifacts.artifact_migration_v9_batch_size

    def handle_artifact_producer_uri(record):
        record.producer_uri = (
            record.full_object.get("spec", {}).get("producer", {}).get("uri", "")
        )
        if record.producer_uri is None:
            record.producer_uri = ""
        return record

    def filter_artifacts():
        return getattr(framework.db.sqldb.models.ArtifactV2, "producer_uri").is_(None)

    _migrate_data(
        db,
        db_session,
        model=framework.db.sqldb.models.ArtifactV2,
        filter_func=filter_artifacts,
        handle_field_record_func=handle_artifact_producer_uri,
        chunk_size=chunk_size,
    )


def _migrate_data(
    db: framework.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    model,
    filter_func,
    handle_field_record_func,
    chunk_size: int = 500,
):
    # Query for records that need migration
    records = db._query(db_session, model).filter(filter_func).limit(chunk_size).all()

    if not records:
        mlrun.utils.logger.info(f"No records to migrate for {model.__name__.lower()}")
        return

    mlrun.utils.logger.info(
        f"Starting migration for {len(records)} {model.__name__.lower()} records"
    )

    while records:
        to_commit = []
        for record in records:
            result = handle_field_record_func(record)
            if isinstance(result, list | tuple | set):
                to_commit.extend(result)
            elif result is not None:
                to_commit.append(result)

        # Commit if there are records to migrate
        if to_commit:
            mlrun.utils.logger.info(
                "Committing migrated records",
                model=model.__name__,
                count=len(to_commit),
            )
            db_session.add_all(to_commit)
            db._commit(db_session, to_commit)

        # Fetch next batch of records to migrate (if any)
        records = (
            db._query(db_session, model).filter(filter_func).limit(chunk_size).all()
        )

        # If no records left to migrate, stop
        if not records:
            mlrun.utils.logger.info("No more records to migrate", model=model.__name__)
            break


def _ensure_latest_tag_for_artifacts(
    db_session: sqlalchemy.orm.Session, chunk_size: int | None = None
):
    chunk_size = chunk_size or mlrun.mlconf.artifacts.artifact_migration_v9_batch_size

    # Note: when logging the same artifact and spawning tags in version < 1.8  and then migrating to 1.8,
    # two artifacts should remain at the end

    # Step 1: Get the latest artifact row for each combination of project, key, and iteration
    subquery = db_session.query(
        framework.db.sqldb.models.ArtifactV2.id,
        framework.db.sqldb.models.ArtifactV2.key,
        framework.db.sqldb.models.ArtifactV2.project,
        framework.db.sqldb.models.ArtifactV2.iteration,
        sqlalchemy.func.row_number()
        .over(
            partition_by=[
                framework.db.sqldb.models.ArtifactV2.project,
                framework.db.sqldb.models.ArtifactV2.key,
                framework.db.sqldb.models.ArtifactV2.iteration,
            ],
            order_by=framework.db.sqldb.models.ArtifactV2.updated.desc(),
        )
        .label("row_number"),
    ).subquery()

    # Step 2: Get only the latest row for each combination of project, key, and iteration
    subquery_filtered = (
        db_session.query(
            subquery.c.id,
            subquery.c.key,
            subquery.c.project,
            subquery.c.iteration,
        )
        .filter(subquery.c.row_number == 1)  # Get only the latest for each combination
        .subquery()
    )

    # Step 3: Query to join with Tag table
    query = db_session.query(
        subquery_filtered.c.id,
        subquery_filtered.c.key,
        subquery_filtered.c.project,
        subquery_filtered.c.iteration,
    ).outerjoin(
        framework.db.sqldb.models.ArtifactV2.Tag,
        framework.db.sqldb.models.ArtifactV2.Tag.obj_id == subquery_filtered.c.id,
    )

    # Step 4: Collect project+key pairs for iteration 0 and >0
    latest_with_iter_0 = query.filter(
        framework.db.sqldb.models.ArtifactV2.Tag.name == "latest",
        subquery_filtered.c.iteration == 0,
    )
    latest_with_iter_gt_0 = query.filter(
        framework.db.sqldb.models.ArtifactV2.Tag.name == "latest",
        subquery_filtered.c.iteration > 0,
    )

    # Collecting the two sets of (project, key) tuples
    project_key_iter_0 = set(
        latest_with_iter_0.with_entities(
            subquery_filtered.c.project, subquery_filtered.c.key
        )
        .distinct()
        .all()
    )

    project_key_iter_gt_0 = set(
        latest_with_iter_gt_0.with_entities(
            subquery_filtered.c.project, subquery_filtered.c.key
        )
        .distinct()
        .all()
    )

    # Create an alias for the Tag table for the NOT EXISTS condition
    tag_alias = sqlalchemy.orm.aliased(framework.db.sqldb.models.ArtifactV2.Tag)

    # Step 5: Collect all artifacts that need to be tagged, filter out artifacts that already have the "latest" tag
    query = query.filter(
        ~sqlalchemy.exists().where(
            sqlalchemy.and_(
                tag_alias.obj_id == subquery_filtered.c.id, tag_alias.name == "latest"
            )
        )
    ).distinct()

    processed_artifacts = set()

    while True:
        # Filter artifacts that have already been processed, as there are artifacts that were processed but not tagged.
        artifacts_to_tag = (
            query.filter(~subquery_filtered.c.id.in_(processed_artifacts))
            .limit(chunk_size)
            .all()
        )

        if not artifacts_to_tag:
            mlrun.utils.logger.info(
                "No artifacts without 'latest' were found",
                model=framework.db.sqldb.models.ArtifactV2.Tag,
            )
            break

        mlrun.utils.logger.info(
            "Starting artifacts without 'latest' tag migration",
            model=framework.db.sqldb.models.ArtifactV2.Tag,
            count=len(artifacts_to_tag),
        )

        new_tags = []
        for artifact_id, key, project, iteration in artifacts_to_tag:
            new_tag = _tag_artifact(
                artifact_id,
                key,
                project,
                iteration,
                project_key_iter_0,
                project_key_iter_gt_0,
            )

            if new_tag:
                new_tags.append(new_tag)

            processed_artifacts.add(artifact_id)

        if new_tags:
            mlrun.utils.logger.info(
                "Committing migrated records",
                model=framework.db.sqldb.models.ArtifactV2.Tag,
                count=len(new_tags),
            )
            db_session.add_all(new_tags)
            db_session.commit()

    mlrun.utils.logger.info("No more artifacts to migrate.")


def _tag_artifact(
    artifact_id, key, project, iteration, project_key_iter_0, project_key_iter_gt_0
):
    """Tags an artifact as 'latest' depending on its iteration and project+key set."""

    # Note: In cases where the same project and key were created from both a hyper-params run and a single run, and the
    # user removed the 'latest' tag from all items, we will assign the 'latest' tag to either the hyper-params items
    # or the single run items. This will depend on which item we encounter first when iterating over the results.

    new_tag = None

    if iteration == 0 and (project, key) not in project_key_iter_gt_0:
        new_tag = framework.db.sqldb.models.ArtifactV2.Tag(
            project=project,
            name="latest",
            obj_id=artifact_id,
            obj_name=key,
        )
        project_key_iter_0.add((project, key))  # Add to iter=0 set
    elif iteration > 0 and (project, key) not in project_key_iter_0:
        new_tag = framework.db.sqldb.models.ArtifactV2.Tag(
            project=project,
            name="latest",
            obj_id=artifact_id,
            obj_name=key,
        )
        project_key_iter_gt_0.add((project, key))  # Add to iter>0 set

    return new_tag


def _create_project_summaries(db, db_session):
    # Create a project summary record for all projects.
    # We need to create them manually because a summary record is created only when a new
    # project is created, so project that existing prior to the upgrade don't have summaries.
    projects = db.list_projects(
        db_session, format_=mlrun.common.formatters.ProjectFormat.name_only
    )
    project_summaries = [
        framework.db.sqldb.models.ProjectSummary(
            project=project_name,
            summary=mlrun.common.schemas.ProjectSummary(name=project_name).dict(),
        )
        for project_name in projects.projects
    ]
    db._upsert(db_session, project_summaries, ignore=True)


def _init_system_id(db_session: sqlalchemy.orm.Session):
    """
    Initializes a system id for MLRun deployment.
    The system id is first checked in the database. If it does not exist, the function checks if an id was set in the
    config, and if neither is found, a new random one is generated and stored.
    """

    db = framework.db.sqldb.db.SQLDB()

    # check if a system id already exists in the database
    system_id = db.get_system_id(db_session)

    if system_id is not None:
        mlrun.utils.logger.debug(
            "Existing system id found in the database", system_id=system_id
        )
        mlrun.mlconf.system_id = system_id
        return

    mlrun.utils.logger.debug("System id not found in DB")
    # check if the system id is already set in the config
    system_id = _get_configured_system_id()

    if system_id:
        mlrun.utils.logger.debug("Using configured system id", system_id=system_id)
    else:
        # if no system id is found, generate a new one
        system_id = _generate_system_id()
    db.store_system_id(db_session, system_id)

    # set the system id in mlrun config
    mlrun.mlconf.system_id = system_id

    mlrun.utils.logger.info("Initialized system ID", system_id=system_id)


def _get_configured_system_id() -> str | None:
    system_id = mlrun.mlconf.system_id or None
    if system_id is None:
        return None

    # UUID has some issues in several system-id use cases (length, hyphens, etc.)
    # so we use only a subset of the UUID.
    try:
        uuid.UUID(system_id)
    except ValueError:
        return system_id

    return system_id.replace("-", "")[: mlrun.mlconf.system_id_len]


def _generate_system_id() -> str:
    # Generate an alphanumeric ID using lowercase letters and digits only
    valid_chars = string.ascii_lowercase + string.digits

    return "".join(random.choices(valid_chars, k=mlrun.mlconf.system_id_len))


def main() -> None:
    init_data()


if __name__ == "__main__":
    main()
