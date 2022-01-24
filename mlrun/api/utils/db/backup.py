import os
import pathlib
import shutil
import subprocess
import typing

import mlrun.api.utils.db.mysql
from mlrun import mlconf
from mlrun.utils import logger


class DBBackupUtil(object):
    def backup_database(self, backup_file_name: str) -> None:
        if ":memory:" in mlconf.httpdb.dsn:
            return
        elif "mysql" in mlconf.httpdb.dsn:
            self._backup_database_mysql(backup_file_name)
        else:
            self._backup_database_sqlite(backup_file_name)

    def load_database_from_backup(
        self, backup_file_name: str, new_backup_file_name: str
    ) -> None:

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
            "mysqldump "
            f"-h f{dsn_data['host']} "
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
            "Loading mysql DB backup data", backup_path=backup_path,
        )
        dsn_data = mlrun.api.utils.db.mysql.MySQLUtil.get_mysql_dsn_data()
        self._run_shell_command(
            "mysql "
            f"-h f{dsn_data['host']} "
            f"-P {dsn_data['port']} "
            f"-u {dsn_data['username']} "
            f"{dsn_data['database']} < {backup_path}"
        )

    def _get_backup_file_path(
        self, backup_file_name: str
    ) -> typing.Optional[pathlib.Path]:
        if ":memory:" in mlconf.httpdb.dsn:
            return
        elif "mysql" in mlconf.httpdb.dsn:
            db_dir_path = pathlib.Path(mlconf.httpdb.dirpath) / "mysql"
        else:
            db_file_path = self._get_sqlite_db_file_path()
            db_dir_path = pathlib.Path(os.path.dirname(db_file_path))
        return db_dir_path / backup_file_name

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
    def _run_shell_command(command):
        logger.debug(
            "Running shell command", command=command,
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            shell=True,
        )
        return process.wait()
