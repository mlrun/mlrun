import typing

import sqlalchemy.orm

import mlrun
import mlrun.api.db.init_db
import mlrun.api.db.sqldb.db
import mlrun.api.db.sqldb.session
import mlrun.api.initial_data
import mlrun.api.schemas
import mlrun.api.utils.singletons.db


def test_add_data_version_empty_db():
    db, db_session = _initialize_db_without_migrations()
    # currently the latest is 1, which is also the value we'll put there if the db is not empty so change it to 3 to
    # differentiate between the two
    original_latest_data_version = mlrun.api.initial_data.latest_data_version
    mlrun.api.initial_data.latest_data_version = "3"
    assert db.get_current_data_version(db_session, raise_on_not_found=False) is None
    mlrun.api.initial_data._add_initial_data(db_session)
    assert (
        db.get_current_data_version(db_session, raise_on_not_found=True)
        == mlrun.api.initial_data.latest_data_version
    )
    mlrun.api.initial_data.latest_data_version = original_latest_data_version


def test_add_data_version_non_empty_db():
    db, db_session = _initialize_db_without_migrations()
    # currently the latest is 1, which is also the value we'll put there if the db is not empty so change it to 3 to
    # differentiate between the two
    original_latest_data_version = mlrun.api.initial_data.latest_data_version
    mlrun.api.initial_data.latest_data_version = "3"

    assert db.get_current_data_version(db_session, raise_on_not_found=False) is None
    # fill db
    db.create_project(
        db_session,
        mlrun.api.schemas.Project(
            metadata=mlrun.api.schemas.ProjectMetadata(name="project-name"),
        ),
    )
    mlrun.api.initial_data._add_initial_data(db_session)
    assert db.get_current_data_version(db_session, raise_on_not_found=True) == "1"
    mlrun.api.initial_data.latest_data_version = original_latest_data_version


def _initialize_db_without_migrations() -> typing.Tuple[
    mlrun.api.db.sqldb.db.SQLDB, sqlalchemy.orm.Session
]:
    dsn = "sqlite:///:memory:?check_same_thread=false"
    mlrun.mlconf.httpdb.dsn = dsn
    mlrun.api.db.sqldb.session._init_engine(dsn)

    mlrun.api.utils.singletons.db.initialize_db()
    db_session = mlrun.api.db.sqldb.session.create_session()
    db = mlrun.api.db.sqldb.db.SQLDB(dsn)
    db.initialize(db_session)
    mlrun.api.db.init_db.init_db(db_session)
    return db, db_session
