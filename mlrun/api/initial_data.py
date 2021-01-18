import collections
import datetime
import os
import pathlib
import sqlalchemy.orm
import typing

from mlrun.api.db.init_db import init_db
import mlrun.api.db.sqldb.db
import mlrun.api.db.sqldb.models
import mlrun.api.schemas
from mlrun.api.db.session import create_session, close_session
from mlrun.utils import logger
from .utils.alembic import AlembicUtil


def init_data(from_scratch: bool = False) -> None:
    logger.info("Creating initial data")

    # run schema migrations on existing DB or create it with alembic
    dir_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    alembic_config_path = dir_path / "alembic.ini"

    alembic_util = AlembicUtil(alembic_config_path)
    alembic_util.init_alembic(from_scratch=from_scratch)

    db_session = create_session()
    try:
        init_db(db_session)
        _perform_data_migrations(db_session)
    finally:
        close_session(db_session)
    logger.info("Initial data created")


def _perform_data_migrations(db_session: sqlalchemy.orm.Session):
    # FileDB is not really a thing anymore, so using SQLDB directly
    db = mlrun.api.db.sqldb.db.SQLDB("")
    logger.info("Performing data migrations")
    _fill_project_state(db, db_session)
    _fix_artifact_tags_duplications(db, db_session)


def _fix_artifact_tags_duplications(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
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


def _fill_project_state(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    projects = db.list_projects(db_session)
    for project in projects.projects:
        changed = False
        if not project.spec.desired_state:
            changed = True
            project.spec.desired_state = mlrun.api.schemas.ProjectState.online
        if not project.status.state:
            changed = True
            project.status.state = project.spec.desired_state
        if changed:
            logger.debug(
                "Found project without state data. Enriching",
                name=project.metadata.name,
            )
            db.store_project(db_session, project.metadata.name, project)


def main() -> None:
    init_data()


if __name__ == "__main__":
    main()
