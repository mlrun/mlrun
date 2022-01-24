import os.path
import pathlib
import shutil
import typing
import unittest.mock

import pytest

import mlrun.api.utils.db.backup
import mlrun.api.utils.db.mysql
from mlrun import mlconf


class Constants:
    sqlite_db_file_path = "test.db"
    backup_file = "backup.db"
    new_backup_file = "new_backup.db"


def test_backup_and_load_sqlite(mock_db_dsn, mock_shutil_copy, mock_is_file_result):
    dsn = f"sqlite:///{Constants.sqlite_db_file_path}"
    mock_db_dsn(dsn)

    db_backup = mlrun.api.utils.db.backup.DBBackupUtil()
    db_backup.backup_database(Constants.backup_file)

    mock_is_file_result(True)
    db_backup.load_database_from_backup(
        Constants.backup_file, Constants.new_backup_file
    )

    copy_calls = [
        # first copy - backup via the `backup_database` call
        unittest.mock.call(
            Constants.sqlite_db_file_path, pathlib.PosixPath(Constants.backup_file)
        ),
        # second copy - via `load_database_from_backup`, backup the current database before loading backup
        unittest.mock.call(
            Constants.sqlite_db_file_path, pathlib.PosixPath(Constants.new_backup_file)
        ),
        # third copy - via `load_database_from_backup`, load the db backup file
        unittest.mock.call(
            pathlib.PosixPath(Constants.backup_file), Constants.sqlite_db_file_path
        ),
    ]
    mock_shutil_copy.assert_has_calls(copy_calls)


def test_backup_and_load_mysql(mock_db_dsn, mock_mysql_util, mock_is_file_result):
    dsn = "mysql://mysql-dsn"
    mock_db_dsn(dsn)

    db_backup = mlrun.api.utils.db.backup.DBBackupUtil()
    db_backup.backup_database(Constants.backup_file)

    mock_is_file_result(True)
    db_backup.load_database_from_backup(
        Constants.backup_file, Constants.new_backup_file
    )

    backup_file_path = f"{mlrun.api.utils.db.backup.DBBackupUtil.mysql_database_dir}/{Constants.backup_file}"
    new_backup_file_path = f"{mlrun.api.utils.db.backup.DBBackupUtil.mysql_database_dir}/{Constants.new_backup_file}"

    mysql_util_calls = [
        # first backup - backup via the `backup_database` call
        unittest.mock.call(pathlib.PosixPath(backup_file_path)),
        # second backup - via `load_database_from_backup`, backup the current database before loading backup
        unittest.mock.call(pathlib.PosixPath(new_backup_file_path)),
    ]
    mock_mysql_util.dump_database_to_file.assert_has_calls(mysql_util_calls)
    mock_mysql_util.load_database_from_file.assert_called_once_with(
        pathlib.PosixPath(backup_file_path),
    )


def test_load_backup_file_does_not_exist_sqlite(
    mock_db_dsn, mock_shutil_copy, mock_is_file_result
):
    dsn = f"sqlite:///{Constants.sqlite_db_file_path}"
    mock_db_dsn(dsn)

    db_backup = mlrun.api.utils.db.backup.DBBackupUtil()

    mock_is_file_result(False)

    with pytest.raises(
        RuntimeError,
        match=f"Cannot load backup from {Constants.backup_file}, file doesn't exist",
    ):
        db_backup.load_database_from_backup(
            Constants.backup_file, Constants.new_backup_file
        )

    mock_shutil_copy.assert_not_called()


def test_load_backup_file_does_not_exist_mysql(
    mock_db_dsn, mock_mysql_util, mock_is_file_result
):
    dsn = "mysql://mysql-dsn"
    mock_db_dsn(dsn)

    db_backup = mlrun.api.utils.db.backup.DBBackupUtil()

    mock_is_file_result(False)

    with pytest.raises(
        RuntimeError,
        match=f"Cannot load backup from {Constants.backup_file}, file doesn't exist",
    ):
        db_backup.load_database_from_backup(
            Constants.backup_file, Constants.new_backup_file
        )

    mock_mysql_util.dump_database_to_file.assert_not_called()
    mock_mysql_util.load_database_from_file.assert_not_called()


@pytest.fixture()
def mock_db_dsn(monkeypatch) -> typing.Callable:
    def _mock_db_dsn(dsn):
        monkeypatch.setattr(mlconf.httpdb, "dsn", dsn)

    return _mock_db_dsn


@pytest.fixture()
def mock_is_file_result(monkeypatch) -> typing.Callable:
    def _mock_is_file_result(result=True):
        def _file_exists(_):
            return result

        monkeypatch.setattr(os.path, "isfile", _file_exists)

    return _mock_is_file_result


@pytest.fixture()
def mock_shutil_copy(monkeypatch) -> unittest.mock.Mock:
    copy = unittest.mock.Mock()
    monkeypatch.setattr(shutil, "copy2", copy)
    return copy


@pytest.fixture()
def mock_mysql_util(monkeypatch) -> unittest.mock.Mock:
    mysql_util = unittest.mock.Mock()

    def _mysql_util():
        return mysql_util

    monkeypatch.setattr(mlrun.api.utils.db.mysql, "MySQLUtil", _mysql_util)
    return mysql_util
