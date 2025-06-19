# Copyright 2025 Iguazio
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
import abc

import sqlalchemy
import sqlalchemy.exc

import mlrun.common.db.dialects


class LockKiller(abc.ABC):
    dialect: None

    def __new__(cls, conn: sqlalchemy.Connection):
        if cls is LockKiller:
            name = conn.dialect.name
            if name.startswith(mlrun.common.db.dialects.Dialects.MYSQL):
                return super().__new__(MySQLLockKiller)
            if name.startswith(mlrun.common.db.dialects.Dialects.POSTGRESQL):
                return super().__new__(PostgresLockKiller)
            raise NotImplementedError(f"No lock killer for dialect '{name}'")
        return super().__new__(cls)

    def __init__(self, conn: sqlalchemy.Connection) -> None:
        self.conn = conn

    @property
    def _log_msg(self) -> str:
        return f"Terminating {self.dialect} session holding lock."

    def kill_locks(self) -> None:
        rows = self.conn.execute(sqlalchemy.text(self.query())).fetchall()
        for row in rows:
            pid, user, addr, obj = self._unpack(row)
            mlrun.utils.logger.warning(
                self._log_msg,
                pid=pid,
                user=user,
                client_addr=addr,
                locked_object=obj,
            )
            self._terminate(pid)

    def _unpack(self, row: sqlalchemy.Row) -> tuple[int, str, str, str]:
        pid, user, addr, obj = row
        return pid, user, addr, obj

    @abc.abstractmethod
    def query(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def _terminate(self, pid: int) -> None:
        raise NotImplementedError()


class MySQLLockKiller(LockKiller):
    dialect = mlrun.common.db.dialects.Dialects.MYSQL

    def query(self) -> str:
        return """
        SELECT
          t.PROCESSLIST_ID,
          t.PROCESSLIST_USER,
          t.PROCESSLIST_HOST,
          GROUP_CONCAT(DISTINCT ml.OBJECT_NAME ORDER BY ml.OBJECT_NAME SEPARATOR ', ')
            AS locked_object
        FROM performance_schema.metadata_locks AS ml
        JOIN performance_schema.threads AS t
          ON ml.OWNER_THREAD_ID = t.THREAD_ID
        WHERE t.PROCESSLIST_ID <> CONNECTION_ID()
          AND ml.OBJECT_SCHEMA = 'mlrun'
          AND ml.OBJECT_NAME != 'alembic_version'
          AND ml.LOCK_STATUS = 'GRANTED'
        GROUP BY t.PROCESSLIST_ID, t.PROCESSLIST_USER, t.PROCESSLIST_HOST
        ORDER BY t.PROCESSLIST_ID;
        """

    def _terminate(self, pid: int) -> None:
        try:
            self.conn.execute(sqlalchemy.text(f"KILL {pid};"))
        except sqlalchemy.exc.OperationalError as e:
            if "Unknown thread id" in str(e):
                mlrun.utils.logger.warning(
                    "DB connection already closed.",
                    pid=pid,
                )
            else:
                raise


class PostgresLockKiller(LockKiller):
    dialect = mlrun.common.db.dialects.Dialects.POSTGRESQL

    def query(self) -> str:
        return """
        SELECT
          a.pid,
          a.usename,
          COALESCE(a.client_addr::text, '') AS client_addr,
          c.relname AS locked_object
        FROM pg_catalog.pg_locks l
        JOIN pg_catalog.pg_class c ON l.relation = c.oid
        JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
        JOIN pg_catalog.pg_stat_activity a ON l.pid = a.pid
        WHERE n.nspname = 'mlrun'
          AND c.relname != 'alembic_version'
          AND l.granted = true
          AND l.pid <> pg_backend_pid()
        ORDER BY a.pid;
        """

    def _terminate(self, pid: int) -> None:
        self.conn.execute(sqlalchemy.text(f"SELECT pg_terminate_backend({pid});"))
