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

import os.path
import pathlib
import re
import shutil
import stat
import typing
import unittest.mock

import pytest

from mlrun import mlconf

import services.api.utils.db.backup


class Constants:
    sqlite_db_file_path = "test.db"
    backup_file = "backup.db"
    new_backup_file = "new_backup.db"

    mysql_dsn = "mysql+pymysql://root@mlrun-db:3306/mlrun"
    mysql_dsn_with_password = "mysql+pymysql://root:s3cret@mlrun-db:3306/mlrun"
    mysql_password = "s3cret"
    mysql_backup_command = (
        "mysqldump --single-transaction --routines --triggers --max_allowed_packet=64000000 "
        "-h mlrun-db -P 3306 -u root mlrun > {0}"
    )
    mysql_load_backup_command = "mysql -h mlrun-db -P 3306 -u root mlrun < {0}"


def test_backup_and_load_sqlite(mock_db_dsn, mock_shutil_copy, mock_is_file_result):
    dsn = f"sqlite:///{Constants.sqlite_db_file_path}"
    mock_db_dsn(dsn)

    db_backup = services.api.utils.db.backup.DBBackupUtil(backup_rotation=False)
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


def test_backup_and_load_mysql(mock_db_dsn, mock_is_file_result):
    mock_db_dsn(Constants.mysql_dsn)

    db_backup = services.api.utils.db.backup.DBBackupUtil(backup_rotation=False)
    db_backup._run_shell_command = unittest.mock.Mock(return_value=0)
    db_backup.backup_database(Constants.backup_file)

    mock_is_file_result(True)
    db_backup.load_database_from_backup(
        Constants.backup_file, Constants.new_backup_file
    )

    backup_file_path = f"{mlconf.httpdb.dirpath}/mysql/{Constants.backup_file}"
    new_backup_file_path = f"{mlconf.httpdb.dirpath}/mysql/{Constants.new_backup_file}"

    run_shell_command_calls = [
        # first backup - backup via the `backup_database` call
        unittest.mock.call(Constants.mysql_backup_command.format(backup_file_path)),
        # second backup - via `load_database_from_backup`, backup the current database before loading backup
        unittest.mock.call(Constants.mysql_backup_command.format(new_backup_file_path)),
        # load from backup
        unittest.mock.call(
            Constants.mysql_load_backup_command.format(backup_file_path)
        ),
    ]
    db_backup._run_shell_command.assert_has_calls(run_shell_command_calls)


def test_backup_and_load_mysql_with_password(mock_db_dsn, mock_is_file_result):
    mock_db_dsn(Constants.mysql_dsn_with_password)

    db_backup = services.api.utils.db.backup.DBBackupUtil(backup_rotation=False)

    # Inspect each invocation while the temp defaults file still exists (the
    # context manager unlinks it as soon as `_run_shell_command` returns).
    captured: list[dict] = []

    def fake_run(command: str) -> int:
        match = re.search(r"--defaults-extra-file=(\S+)", command)
        assert match is not None, f"command missing --defaults-extra-file: {command}"
        path = match.group(1)
        with open(path) as f:
            content = f.read()
        mode = stat.S_IMODE(os.stat(path).st_mode)
        captured.append(
            {"command": command, "path": path, "content": content, "mode": mode}
        )
        return 0

    db_backup._run_shell_command = unittest.mock.Mock(side_effect=fake_run)
    db_backup.backup_database(Constants.backup_file)

    mock_is_file_result(True)
    db_backup.load_database_from_backup(
        Constants.backup_file, Constants.new_backup_file
    )

    assert db_backup._run_shell_command.call_count == 3
    expected_content = f'[client]\npassword="{Constants.mysql_password}"\n'
    for entry in captured:
        # Password must never appear in the shell command (which is logged).
        assert Constants.mysql_password not in entry["command"]
        # --defaults-extra-file must come first, immediately after the binary.
        assert re.match(r"^(mysqldump|mysql) --defaults-extra-file=", entry["command"])
        # File must be 0600 and contain only the [client] password section.
        assert entry["mode"] == 0o600
        assert entry["content"] == expected_content
        # File must be cleaned up after the command returns.
        assert not os.path.exists(entry["path"])


@pytest.mark.parametrize(
    "password,expected_value_in_file",
    [
        # Backslash → escaped to a double backslash
        ("a\\b", "a\\\\b"),
        # Double-quote → escaped to backslash-quote (terminator-safe)
        ('a"b', 'a\\"b'),
        # Both together — order-sensitive: backslash must be escaped first
        # so we don't double-escape the backslash we just wrote in front of "
        ('p\\a"ss', 'p\\\\a\\"ss'),
        # Plain ASCII passes through untouched
        ("plain", "plain"),
    ],
)
def test_mysql_defaults_file_escapes_special_chars(
    password: str, expected_value_in_file: str
) -> None:
    cm = services.api.utils.db.backup.DBBackupUtil._mysql_defaults_file(
        {"password": password}
    )
    with cm as defaults_arg:
        path = defaults_arg.removeprefix("--defaults-extra-file=").rstrip()
        with open(path) as f:
            content = f.read()
    assert content == f'[client]\npassword="{expected_value_in_file}"\n'


def test_load_backup_file_does_not_exist_sqlite(
    mock_db_dsn, mock_shutil_copy, mock_is_file_result
):
    dsn = f"sqlite:///{Constants.sqlite_db_file_path}"
    mock_db_dsn(dsn)

    db_backup = services.api.utils.db.backup.DBBackupUtil(backup_rotation=False)

    mock_is_file_result(False)

    with pytest.raises(
        RuntimeError,
        match=f"Cannot load backup from {Constants.backup_file}, file doesn't exist",
    ):
        db_backup.load_database_from_backup(
            Constants.backup_file, Constants.new_backup_file
        )

    mock_shutil_copy.assert_not_called()


def test_load_backup_file_does_not_exist_mysql(mock_db_dsn, mock_is_file_result):
    mock_db_dsn(Constants.mysql_dsn)

    db_backup = services.api.utils.db.backup.DBBackupUtil(backup_rotation=False)
    db_backup._run_shell_command = unittest.mock.Mock(return_value=0)

    mock_is_file_result(False)

    with pytest.raises(
        RuntimeError,
        match=f"Cannot load backup from {Constants.backup_file}, file doesn't exist",
    ):
        db_backup.load_database_from_backup(
            Constants.backup_file, Constants.new_backup_file
        )

    db_backup._run_shell_command.assert_not_called()


def test_backup_file_rotation(mock_db_dsn, mock_listdir_result, mock_os_remove):
    mock_db_dsn(Constants.mysql_dsn)

    db_backup = services.api.utils.db.backup.DBBackupUtil(
        backup_rotation=True, backup_rotation_limit=3
    )

    existing_backup_files = [
        f"db_backup_2022012510{minute}.db" for minute in [10, 11, 12, 13]
    ]
    mock_listdir_result(existing_backup_files)
    db_backup._rotate_backup()

    mock_os_remove.assert_called_once_with(
        pathlib.Path(mlconf.httpdb.dirpath) / "mysql" / existing_backup_files[0]
    )


@pytest.fixture()
def mock_db_dsn(monkeypatch) -> typing.Callable:
    dsn_env_var = "MLRUN_HTTPDB__DSN"
    old_dsn_value = os.environ.get(dsn_env_var, None)

    def _mock_db_dsn(dsn):
        monkeypatch.setattr(mlconf.httpdb, "dsn", dsn)
        os.environ[dsn_env_var] = dsn

    yield _mock_db_dsn

    if old_dsn_value is None:
        os.environ.pop(dsn_env_var)
    else:
        os.environ[dsn_env_var] = old_dsn_value


@pytest.fixture()
def mock_is_file_result(monkeypatch) -> typing.Callable:
    def _mock_is_file_result(result=True):
        def _file_exists(_):
            return result

        monkeypatch.setattr(os.path, "isfile", _file_exists)

    return _mock_is_file_result


@pytest.fixture()
def mock_listdir_result(monkeypatch) -> typing.Callable:
    def _mock_listdir_result(result):
        def _listdir(_):
            return result

        monkeypatch.setattr(os, "listdir", _listdir)

    return _mock_listdir_result


@pytest.fixture()
def mock_os_remove(monkeypatch) -> unittest.mock.Mock:
    remove = unittest.mock.Mock()
    monkeypatch.setattr(os, "remove", remove)
    return remove


@pytest.fixture()
def mock_shutil_copy(monkeypatch) -> unittest.mock.Mock:
    copy = unittest.mock.Mock()
    monkeypatch.setattr(shutil, "copy2", copy)
    return copy
