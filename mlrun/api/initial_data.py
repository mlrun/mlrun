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
import collections
import datetime
import json
import os
import pathlib
import typing

import dateutil.parser
import pydantic.error_wrappers
import pymysql.err
import sqlalchemy.exc
import sqlalchemy.orm

import mlrun.api.db.sqldb.db
import mlrun.api.db.sqldb.helpers
import mlrun.api.db.sqldb.models
import mlrun.api.utils.db.alembic
import mlrun.api.utils.db.backup
import mlrun.api.utils.db.mysql
import mlrun.api.utils.db.sqlite_migration
import mlrun.artifacts
import mlrun.artifacts.base
import mlrun.common.schemas
from mlrun.api.db.init_db import init_db
from mlrun.api.db.session import close_session, create_session
from mlrun.config import config
from mlrun.errors import err_to_str
from mlrun.utils import (
    fill_artifact_object_hash,
    is_legacy_artifact,
    is_link_artifact,
    logger,
)


def init_data(
    from_scratch: bool = False, perform_migrations_if_needed: bool = False
) -> None:
    logger.info("Initializing DB data")

    # create mysql util, and if mlrun is configured to use mysql, wait for it to be live and set its db modes
    mysql_util = mlrun.api.utils.db.mysql.MySQLUtil(logger)
    if mysql_util.get_mysql_dsn_data():
        mysql_util.wait_for_db_liveness()
        mysql_util.set_modes(mlrun.mlconf.httpdb.db.mysql.modes)
    else:
        dsn = mysql_util.get_dsn()
        if "sqlite" in dsn:
            logger.debug("SQLite DB is used, liveness check not needed")
        else:
            logger.warn(
                f"Invalid mysql dsn: {dsn}, assuming live and skipping liveness verification"
            )

    sqlite_migration_util = None
    if not from_scratch and config.httpdb.db.database_migration_mode == "enabled":
        sqlite_migration_util = (
            mlrun.api.utils.db.sqlite_migration.SQLiteMigrationUtil()
        )
    alembic_util = _create_alembic_util()
    (
        is_migration_needed,
        is_migration_from_scratch,
        is_backup_needed,
    ) = _resolve_needed_operations(alembic_util, sqlite_migration_util, from_scratch)

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
        db_backup = mlrun.api.utils.db.backup.DBBackupUtil()
        db_backup.backup_database()

    logger.info("Creating initial data")
    config.httpdb.state = mlrun.common.schemas.APIStates.migrations_in_progress

    if is_migration_from_scratch or is_migration_needed:
        try:
            _perform_schema_migrations(alembic_util)

            _perform_database_migration(sqlite_migration_util)

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
    if not is_migration_from_scratch and is_migration_needed:
        config.httpdb.state = mlrun.common.schemas.APIStates.migrations_completed
    else:
        config.httpdb.state = mlrun.common.schemas.APIStates.online
    logger.info("Initial data created")


# If the data_table version doesn't exist, we can assume the data version is 1.
# This is because data version 1 points to a data migration which was added back in 0.6.0, and
# upgrading from a version earlier than 0.6.0 to v>=0.8.0 is not supported.
data_version_prior_to_table_addition = 1

# NOTE: Bump this number when adding a new data migration
latest_data_version = 4


def _resolve_needed_operations(
    alembic_util: mlrun.api.utils.db.alembic.AlembicUtil,
    sqlite_migration_util: typing.Optional[
        mlrun.api.utils.db.sqlite_migration.SQLiteMigrationUtil
    ],
    force_from_scratch: bool = False,
) -> typing.Tuple[bool, bool, bool]:
    is_database_migration_needed = False
    if sqlite_migration_util is not None:
        is_database_migration_needed = (
            sqlite_migration_util.is_database_migration_needed()
        )
    # the util checks whether the target DB has data, when database migration needed, it obviously does not have data
    # but in that case it's not really a migration from scratch
    is_migration_from_scratch = (
        force_from_scratch or alembic_util.is_migration_from_scratch()
    ) and not is_database_migration_needed
    is_schema_migration_needed = alembic_util.is_schema_migration_needed()
    is_data_migration_needed = (
        not _is_latest_data_version()
        and config.httpdb.db.data_migrations_mode == "enabled"
    )
    is_migration_needed = is_database_migration_needed or (
        not is_migration_from_scratch
        and (is_schema_migration_needed or is_data_migration_needed)
    )
    is_backup_needed = (
        config.httpdb.db.backup.mode == "enabled"
        and is_migration_needed
        and not is_migration_from_scratch
        and not is_database_migration_needed
    )
    logger.info(
        "Checking if migration is needed",
        is_migration_from_scratch=is_migration_from_scratch,
        is_schema_migration_needed=is_schema_migration_needed,
        is_data_migration_needed=is_data_migration_needed,
        is_database_migration_needed=is_database_migration_needed,
        is_backup_needed=is_backup_needed,
        is_migration_needed=is_migration_needed,
    )

    return is_migration_needed, is_migration_from_scratch, is_backup_needed


def _create_alembic_util() -> mlrun.api.utils.db.alembic.AlembicUtil:
    alembic_config_file_name = "alembic.ini"
    if mlrun.api.utils.db.mysql.MySQLUtil.get_mysql_dsn_data():
        alembic_config_file_name = "alembic_mysql.ini"

    # run schema migrations on existing DB or create it with alembic
    dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    alembic_config_path = dir_path / alembic_config_file_name

    alembic_util = mlrun.api.utils.db.alembic.AlembicUtil(
        alembic_config_path, _is_latest_data_version()
    )
    return alembic_util


def _perform_schema_migrations(alembic_util: mlrun.api.utils.db.alembic.AlembicUtil):
    logger.info("Performing schema migration")
    alembic_util.init_alembic()


def _is_latest_data_version():
    db_session = create_session()
    db = mlrun.api.db.sqldb.db.SQLDB("")

    try:
        current_data_version = _resolve_current_data_version(db, db_session)
    finally:
        close_session(db_session)

    return current_data_version == latest_data_version


def _perform_database_migration(
    sqlite_migration_util: typing.Optional[
        mlrun.api.utils.db.sqlite_migration.SQLiteMigrationUtil
    ],
):
    if sqlite_migration_util:
        logger.info("Performing database migration")
        sqlite_migration_util.transfer()


def _perform_data_migrations(db_session: sqlalchemy.orm.Session):
    if config.httpdb.db.data_migrations_mode == "enabled":
        # FileDB is not really a thing anymore, so using SQLDB directly
        db = mlrun.api.db.sqldb.db.SQLDB("")
        current_data_version = int(db.get_current_data_version(db_session))
        if current_data_version != latest_data_version:
            logger.info(
                "Performing data migrations",
                current_data_version=current_data_version,
                latest_data_version=latest_data_version,
            )
            if current_data_version < 1:
                _perform_version_1_data_migrations(db, db_session)
            if current_data_version < 2:
                _perform_version_2_data_migrations(db, db_session)
            if current_data_version < 3:
                _perform_version_3_data_migrations(db, db_session)
            if current_data_version < 4:
                _perform_version_4_data_migrations(db, db_session)

            db.create_data_version(db_session, str(latest_data_version))


def _add_initial_data(db_session: sqlalchemy.orm.Session):
    # FileDB is not really a thing anymore, so using SQLDB directly
    db = mlrun.api.db.sqldb.db.SQLDB("")
    _add_default_hub_source_if_needed(db, db_session)
    _add_data_version(db, db_session)


def _perform_version_1_data_migrations(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _enrich_project_state(db, db_session)
    _fix_artifact_tags_duplications(db, db_session)
    _fix_datasets_large_previews(db, db_session)


def _enrich_project_state(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    logger.info("Enriching projects state")
    projects = db.list_projects(db_session)
    for project in projects.projects:
        changed = False
        if not project.spec.desired_state:
            changed = True
            project.spec.desired_state = mlrun.common.schemas.ProjectState.online
        if not project.status.state:
            changed = True
            project.status.state = project.spec.desired_state
        if changed:
            logger.debug(
                "Found project without state data. Enriching",
                name=project.metadata.name,
            )
            db.store_project(db_session, project.metadata.name, project)


def _fix_artifact_tags_duplications(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    logger.info("Fixing artifact tags duplications")
    # get all artifacts
    artifacts = db._find_artifacts(db_session, None, "*")
    # get all artifact tags
    tags = db._query(db_session, mlrun.api.db.sqldb.models.Artifact.Tag).all()
    # artifact record id -> artifact
    artifact_record_id_map = {artifact.id: artifact for artifact in artifacts}
    tags_to_delete = []
    projects = {artifact.project for artifact in artifacts}
    for project in projects:
        artifact_keys = {
            artifact.key for artifact in artifacts if artifact.project == project
        }
        for artifact_key in artifact_keys:
            artifact_key_tags = []
            for tag in tags:
                # sanity
                if tag.obj_id not in artifact_record_id_map:
                    logger.warning("Found orphan tag, deleting", tag=tag.to_dict())
                if artifact_record_id_map[tag.obj_id].key == artifact_key:
                    artifact_key_tags.append(tag)
            tag_name_tags_map = collections.defaultdict(list)
            for tag in artifact_key_tags:
                tag_name_tags_map[tag.name].append(tag)
            for tag_name, _tags in tag_name_tags_map.items():
                if len(_tags) == 1:
                    continue
                tags_artifacts = [artifact_record_id_map[tag.obj_id] for tag in _tags]
                last_updated_artifact = _find_last_updated_artifact(tags_artifacts)
                for tag in _tags:
                    if tag.obj_id != last_updated_artifact.id:
                        tags_to_delete.append(tag)
    if tags_to_delete:
        logger.info(
            "Found duplicated artifact tags. Removing duplications",
            tags_to_delete=[
                tag_to_delete.to_dict() for tag_to_delete in tags_to_delete
            ],
            tags=[tag.to_dict() for tag in tags],
            artifacts=[artifact.to_dict() for artifact in artifacts],
        )
        for tag in tags_to_delete:
            db_session.delete(tag)
        db_session.commit()


def _fix_datasets_large_previews(
    db: mlrun.api.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
):
    logger.info("Fixing datasets large previews")
    # get all artifacts
    artifacts = db._find_artifacts(db_session, None, "*")
    for artifact in artifacts:
        try:
            artifact_dict = artifact.struct
            if (
                artifact_dict
                and artifact_dict.get("kind") == mlrun.artifacts.DatasetArtifact.kind
            ):
                is_legacy = is_legacy_artifact(artifact_dict)

                header = (
                    artifact_dict.get("header", [])
                    if is_legacy
                    else artifact_dict.get("spec", {}).get("header", [])
                )
                if header and len(header) > mlrun.artifacts.dataset.max_preview_columns:
                    logger.debug(
                        "Found dataset artifact with more than allowed columns in preview fields. Fixing",
                        artifact=artifact_dict,
                    )
                    columns_to_remove = header[
                        mlrun.artifacts.dataset.max_preview_columns :
                    ]

                    # align preview
                    preview = (
                        artifact_dict.get("preview")
                        if is_legacy
                        else artifact_dict.get("status", {}).get("preview")
                    )
                    if preview:
                        new_preview = []
                        for preview_row in preview:
                            # sanity
                            if (
                                len(preview_row)
                                < mlrun.artifacts.dataset.max_preview_columns
                            ):
                                logger.warning(
                                    "Found artifact with more than allowed columns in header definition, "
                                    "but preview data is valid. Leaving preview as is",
                                    artifact=artifact_dict,
                                )
                            new_preview.append(
                                preview_row[
                                    : mlrun.artifacts.dataset.max_preview_columns
                                ]
                            )

                        if is_legacy:
                            artifact_dict["preview"] = new_preview
                        else:
                            artifact_dict["status"]["preview"] = new_preview

                    # align stats
                    for column_to_remove in columns_to_remove:
                        artifact_stats = (
                            artifact_dict.get("stats", {})
                            if is_legacy
                            else artifact_dict.get("status").get("stats", {})
                        )
                        if column_to_remove in artifact_stats:
                            del artifact_stats[column_to_remove]

                    # align schema
                    schema_dict = (
                        artifact_dict.get("schema", {})
                        if is_legacy
                        else artifact_dict.get("spec").get("schema", {})
                    )
                    if schema_dict.get("fields"):
                        new_schema_fields = []
                        for field in schema_dict["fields"]:
                            if field.get("name") not in columns_to_remove:
                                new_schema_fields.append(field)
                        schema_dict["fields"] = new_schema_fields

                    # lastly, align headers
                    if is_legacy:
                        artifact_dict["header"] = header[
                            : mlrun.artifacts.dataset.max_preview_columns
                        ]
                    else:
                        artifact_dict["spec"]["header"] = header[
                            : mlrun.artifacts.dataset.max_preview_columns
                        ]

                    logger.debug(
                        "Fixed dataset artifact preview fields. Storing",
                        artifact=artifact_dict,
                    )
                    db._store_artifact(
                        db_session,
                        artifact.key,
                        artifact_dict,
                        artifact.uid,
                        project=artifact.project,
                        tag_artifact=False,
                    )
        except Exception as exc:
            logger.warning(
                "Failed fixing dataset artifact large preview. Continuing",
                exc=exc,
            )


def _find_last_updated_artifact(
    artifacts: typing.List[mlrun.api.db.sqldb.models.Artifact],
):
    # sanity
    if not artifacts:
        raise RuntimeError("No artifacts given")
    last_updated_artifact = None
    last_updated_artifact_time = datetime.datetime.min
    artifacts_with_same_update_time = []
    for artifact in artifacts:
        if artifact.updated > last_updated_artifact_time:
            last_updated_artifact = artifact
            last_updated_artifact_time = last_updated_artifact.updated
            artifacts_with_same_update_time = [last_updated_artifact]
        elif artifact.updated == last_updated_artifact_time:
            artifacts_with_same_update_time.append(artifact)
    if len(artifacts_with_same_update_time) > 1:
        logger.warning(
            "Found several artifact with same update time, heuristically choosing the first",
            artifacts=[
                artifact.to_dict() for artifact in artifacts_with_same_update_time
            ],
        )
        # we don't really need to do anything to choose the first, it's already happening because the first if is >
        # and not >=
    if not last_updated_artifact:
        logger.warning(
            "No artifact had update time, heuristically choosing the first",
            artifacts=[artifact.to_dict() for artifact in artifacts],
        )
        last_updated_artifact = artifacts[0]

    return last_updated_artifact


def _perform_version_2_data_migrations(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _align_runs_table(db, db_session)


def _align_runs_table(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    logger.info("Aligning runs")
    runs = db._find_runs(db_session, None, "*", None).all()
    for run in runs:
        run_dict = run.struct

        # Align run start_time column to the start time from the body
        run.start_time = (
            mlrun.api.db.sqldb.helpers.run_start_time(run_dict) or run.start_time
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
            "state", mlrun.runtimes.constants.RunStates.created
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
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _rename_marketplace_kind_to_hub(db, db_session)


def _rename_marketplace_kind_to_hub(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
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
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    _migrate_artifacts_table_v2(db, db_session)


def _migrate_artifacts_table_v2(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    """
    Migrate the old artifacts table to the new artifacts_v2 table, including their respective tags and labels.
    The migration is done in batches, to not overload the db. A state file is used to keep track of the migration
    progress, and is updated after each batch, so that if the migration fails, it can be resumed from the last batch.
    Delete the old artifacts table when done.
    """
    logger.info("Migrating artifacts to artifacts_v2 table")

    # count the total number of artifacts to migrate
    total_artifacts_count = db._query(
        db_session, mlrun.api.db.sqldb.models.Artifact
    ).count()
    batch_size = config.artifacts.artifact_migration_batch_size

    # get the id of the last migrated artifact and the list of all link artifacts ids from the state file
    last_migrated_artifact_id, link_artifact_ids = _get_migration_state()

    while True:
        logger.debug(
            "Migrating artifacts batch",
            batch_size=batch_size,
            total_artifacts_count=total_artifacts_count,
        )
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

    # drop the old artifacts table, including their labels and tags tables
    db.delete_table_records(
        db_session, mlrun.api.db.sqldb.models.Artifact.Label, raise_on_not_exists=False
    )
    db.delete_table_records(
        db_session, mlrun.api.db.sqldb.models.Artifact.Tag, raise_on_not_exists=False
    )
    db.delete_table_records(
        db_session, mlrun.api.db.sqldb.models.Artifact, raise_on_not_exists=False
    )


def _migrate_artifacts_batch(
    db: mlrun.api.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    last_migrated_artifact_id: int,
    batch_size: int,
):
    new_artifacts = []
    artifacts_tags_to_migrate = []
    artifacts_labels_to_migrate = []
    link_artifact_ids = []

    # get artifacts from the db, sorted by id
    query = db._query(db_session, mlrun.api.db.sqldb.models.Artifact)
    if last_migrated_artifact_id > 0:
        # skip the artifacts that were already migrated
        query = query.filter(
            mlrun.api.db.sqldb.models.Artifact.id > last_migrated_artifact_id
        )

    query = query.order_by(mlrun.api.db.sqldb.models.Artifact.id).limit(batch_size)

    artifacts = query.all()

    if len(artifacts) == 0:
        # we're done
        return None, None

    for artifact in artifacts:
        new_artifact = mlrun.api.db.sqldb.models.ArtifactV2()

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

        artifact_metadata = artifact_dict.get("metadata", None)

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

        # key - the artifact's key, without iteration if it is attached to it
        key = artifact_metadata.get("key", "")
        new_artifact.key = key

        # iteration - the artifact's iteration
        iteration = artifact_metadata.get("iter", None)
        if iteration is not None:
            new_artifact.iteration = int(iteration)

        # best iteration
        # if iteration == 0 it means it is from a single run since link artifacts were already
        # handled above - so we can set is as best iteration.
        # otherwise set to false, the best iteration artifact will be updated later
        if iteration is not None and iteration == 0:
            new_artifact.best_iteration = True
        else:
            new_artifact.best_iteration = False

        # uid - calculate as the hash of the artifact object
        uid = fill_artifact_object_hash(artifact_dict, "uid", iteration)
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

        # save the artifact's tags and labels to migrate them later
        tag = artifact_metadata.get("tag", "")
        if tag:
            artifacts_tags_to_migrate.append((new_artifact, tag))
        labels = artifact_metadata.get("labels", {})
        if labels:
            artifacts_labels_to_migrate.append((new_artifact, labels))

    # add the new artifacts to the db session
    db_session.add_all(new_artifacts)

    # migrate artifact labels to the new table ("artifact_v2_labels")
    new_labels = _migrate_artifact_labels(db_session, artifacts_labels_to_migrate)

    # migrate artifact tags to the new table ("artifact_v2_tags")
    new_tags = _migrate_artifact_tags(db_session, artifacts_tags_to_migrate)

    # commit the changes
    db._commit(db_session, new_artifacts + new_labels + new_tags)

    return last_migrated_artifact_id, link_artifact_ids


def _migrate_artifact_labels(
    db_session: sqlalchemy.orm.Session,
    artifacts_labels_to_migrate: list,
):
    # iterate over all the artifacts, and create labels for each one
    logger.info("Aligning artifact labels")
    labels = []
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
    artifacts_tags_to_migrate: list,
):
    # iterate over all the artifacts, and create a new tag for each one
    logger.info("Aligning artifact tags")
    tags = []
    for artifact, tag in artifacts_tags_to_migrate:
        if tag:
            new_tag = artifact.Tag(
                project=artifact.project,
                name=tag,
                obj_name=artifact.key,
                obj_id=artifact.id,
            )
            tags.append(new_tag)
    if tags:
        db_session.add_all(tags)
    return tags


def _mark_best_iteration_artifacts(
    db: mlrun.api.db.sqldb.db.SQLDB,
    db_session: sqlalchemy.orm.Session,
    link_artifact_ids: list,
):
    artifacts_to_commit = []

    # get all link artifacts
    link_artifacts = (
        db_session.query(mlrun.api.db.sqldb.models.Artifact)
        .filter(mlrun.api.db.sqldb.models.Artifact.id.in_(link_artifact_ids))
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
        query = db._query(db_session, mlrun.api.db.sqldb.models.ArtifactV2).filter(
            mlrun.api.db.sqldb.models.ArtifactV2.key == link_artifact_key,
            mlrun.api.db.sqldb.models.ArtifactV2.iteration == link_iteration,
        )
        if link_tree:
            query = query.filter(
                mlrun.api.db.sqldb.models.ArtifactV2.producer_id == link_tree
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


def _add_default_hub_source_if_needed(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    try:
        hub_marketplace_source = db.get_hub_source(
            db_session, config.hub.default_source.name
        )
    except mlrun.errors.MLRunNotFoundError:
        hub_marketplace_source = None
    except pydantic.error_wrappers.ValidationError as exc:

        # following the renaming of 'marketplace' to 'hub', validation errors can occur on the old 'marketplace'.
        # this will be handled later in the data migrations, but for now - if a validation error occurs, we assume
        # that a default hub source exists
        if all(
            [
                "validation error for HubSource" in str(exc),
                "value is not a valid enumeration member" in str(exc),
            ]
        ):
            logger.info("Found existing default hub source, data migration needed")
            hub_marketplace_source = True
        else:
            raise exc

    if not hub_marketplace_source:
        hub_source = mlrun.common.schemas.HubSource.generate_default_source()
        # hub_source will be None if the configuration has hub.default_source.create=False
        if hub_source:
            logger.info("Adding default hub source")
            # Not using db.store_marketplace_source() since it doesn't allow changing the default hub source.
            hub_record = db._transform_hub_source_schema_to_record(
                mlrun.common.schemas.IndexedHubSource(
                    index=mlrun.common.schemas.hub.last_source_index,
                    source=hub_source,
                )
            )
            db_session.add(hub_record)
            db_session.commit()
        else:
            logger.info("Not adding default hub source, per configuration")
    return


def _add_data_version(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    if db.get_current_data_version(db_session, raise_on_not_found=False) is None:
        data_version = _resolve_current_data_version(db, db_session)
        logger.info(
            "No data version, setting data version",
            data_version=data_version,
        )
        db.create_data_version(db_session, data_version)


def _resolve_current_data_version(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
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
            projects = db.list_projects(db_session)
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
                exc=exc,
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
                exc=exc,
                data_version_prior_to_table_addition=data_version_prior_to_table_addition,
            )
            return data_version_prior_to_table_addition

        raise exc


def _get_migration_state():
    """
    Get the id of the last migrated artifact from the state file.
    If the state file does not exist, return 0.
    """
    try:
        with open(
            config.artifacts.artifact_migration_state_file_path, "r"
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


def main() -> None:
    init_data()


if __name__ == "__main__":
    main()
