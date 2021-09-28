import pytest
import sqlalchemy.orm

import mlrun
import mlrun.api.db.sqldb.db
import mlrun.errors
import mlrun.projects.project


def test_set_environment_with_invalid_project_name(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    invalid_name = "project_name"
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.set_environment(project=invalid_name)
