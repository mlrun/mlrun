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
import datetime
import os
import pathlib
import shutil
import subprocess
import typing

import mlrun.api.utils.db.mysql
from mlrun import mlconf
from mlrun.utils import logger


class DBBackupUtil(object):
    def __init__(
        self,
        backup_file_format: str = mlconf.httpdb.db.backup.file_format,
        backup_rotation: bool = mlconf.httpdb.db.backup.use_rotation,
        backup_rotation_limit: int = mlconf.httpdb.db.backup.rotation_limit,
    ) -> None:
        self._backup_file_format = backup_file_format
        self._backup_rotation = backup_rotation
        self._backup_rotation_limit = backup_rotation_limit

    def backup_database(self, backup_file_name: str = None) -> None:
        backup_file_name = backup_file_name or self._generate_backup_file_name()

        # ensure the backup directory exists
        self._get_db_dir_path().mkdir(parents=True, exist_ok=True)

        if ":memory:" in mlconf.httpdb.dsn:
            return
        elif "mysql" in mlconf.httpdb.dsn:
            self._backup_database_mysql(backup_file_name)
        else:
            self._backup_database_sqlite(backup_file_name)

        if self._backup_rotation:
            self._rotate_backup()

    def load_database_from_backup(
        self, backup_file_name: str, new_backup_file_name: str = None
    ) -> None:
        new_backup_file_name = new_backup_file_name or self._generate_backup_file_name()

        backup_path = self._get_backup_file_path(backup_file_name)
        if not backup_path or not os.path.isfile(backup_path):
            raise RuntimeError(
                f"Cannot load backup from {backup_file_name}, file doesn't exist"
            )

        # backup the current DB
        self.backup_database(new_backup_file_name)

        if ":memory:" in mlconf.httpdb.dsn:
            return
        elif "mysql" in mlconf.httpdb.dsn:
            self._load_database_backup_mysql(backup_file_name)
        else:
            self._load_database_backup_sqlite(backup_file_name)

    def _backup_database_sqlite(self, backup_file_name: str) -> None:
        db_file_path = self._get_sqlite_db_file_path()
        backup_path = self._get_backup_file_path(backup_file_name)

        logger.debug(
            "Backing up sqlite DB file",
            db_file_path=db_file_path,
            backup_path=backup_path,
        )
        shutil.copy2(db_file_path, backup_path)

    def _load_database_backup_sqlite(self, backup_file_name: str) -> None:
        db_file_path = self._get_sqlite_db_file_path()
        backup_path = self._get_backup_file_path(backup_file_name)

        logger.debug(
            "Loading sqlite DB backup file",
            db_file_path=db_file_path,
            backup_path=backup_path,
        )
        shutil.copy2(backup_path, db_file_path)

    def _backup_database_mysql(self, backup_file_name: str) -> None:
        backup_path = self._get_backup_file_path(backup_file_name)

        logger.debug("Backing up mysql DB data", backup_path=backup_path)
        dsn_data = mlrun.api.utils.db.mysql.MySQLUtil.get_mysql_dsn_data()
        self._run_shell_command(
            "mysqldump --single-transaction --routines --triggers "
            f"--max_allowed_packet={mlconf.httpdb.db.backup.max_allowed_packet} "
            f"-h {dsn_data['host']} "
            f"-P {dsn_data['port']} "
            f"-u {dsn_data['username']} "
            f"{dsn_data['database']} > {backup_path}"
        )

    def _load_database_backup_mysql(self, backup_file_name: str) -> None:
        """
        To run this operation manually, you can either run the command below from the mlrun-api pod or
        enter the mysql pod and run:
        mysql -S /var/run/mysqld/mysql.sock -p mlrun < FILE_PATH
        """
        backup_path = self._get_backup_file_path(backup_file_name)

        logger.debug(
            "Loading mysql DB backup data",
            backup_path=backup_path,
        )
        dsn_data = mlrun.api.utils.db.mysql.MySQLUtil.get_mysql_dsn_data()
        self._run_shell_command(
            "mysql "
            f"-h {dsn_data['host']} "
            f"-P {dsn_data['port']} "
            f"-u {dsn_data['username']} "
            f"{dsn_data['database']} < {backup_path}"
        )

    def _rotate_backup(self) -> None:
        db_dir_path = self._get_db_dir_path()
        dir_content = os.listdir(db_dir_path)
        backup_files = []
        for file_name in dir_content:
            try:
                date_metadata = datetime.datetime.strptime(
                    file_name, self._backup_file_format
                )
            except ValueError:
                continue

            backup_files.append((file_name, date_metadata))

        if len(backup_files) <= self._backup_rotation_limit:
            return

        backup_files = sorted(backup_files, key=lambda file_data: file_data[1])
        files_to_delete = [
            file_data[0] for file_data in backup_files[: -self._backup_rotation_limit]
        ]
        logger.debug("Rotating old backup files", files_to_delete=files_to_delete)
        for file_name in files_to_delete:
            try:
                os.remove(db_dir_path / file_name)
            except FileNotFoundError:
                logger.debug(
                    "Backup file doesn't exist, skipping...", file_name=file_name
                )

    def _generate_backup_file_name(self) -> str:
        return datetime.datetime.now(tz=datetime.timezone.utc).strftime(
            self._backup_file_format
        )

    def _get_backup_file_path(
        self, backup_file_name: str
    ) -> typing.Optional[pathlib.Path]:
        if ":memory:" in mlconf.httpdb.dsn:
            return

        return self._get_db_dir_path() / backup_file_name

    def _get_db_dir_path(self) -> typing.Optional[pathlib.Path]:
        if ":memory:" in mlconf.httpdb.dsn:
            return
        elif "mysql" in mlconf.httpdb.dsn:
            db_dir_path = pathlib.Path(mlconf.httpdb.dirpath) / "mysql"
        else:
            db_file_path = self._get_sqlite_db_file_path()
            db_dir_path = pathlib.Path(os.path.dirname(db_file_path))
        return db_dir_path

    @staticmethod
    def _get_sqlite_db_file_path() -> str:
        """
        Get the db file path from the dsn.
        Converts the dsn to the file path. e.g.:
        sqlite:////mlrun/db/mlrun.db?check_same_thread=false -> /mlrun/db/mlrun.db
        if mysql is used returns empty string
        """
        return mlconf.httpdb.dsn.split("?")[0].split("sqlite:///")[-1]

    @staticmethod
    def _run_shell_command(command: str) -> int:
        logger.debug(
            "Running shell command",
            command=command,
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            shell=True,
        )
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        return_code = process.wait()

        if return_code != 0:
            logger.error(
                "Failed running shell command",
                command=command,
                stdout=stdout,
                stderr=stderr,
                exit_status=return_code,
            )
            raise RuntimeError(
                f"Got non-zero return code ({return_code}) on running shell command: {command}"
            )

        logger.debug(
            "Ran command successfully",
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_status=return_code,
        )

        return return_code
