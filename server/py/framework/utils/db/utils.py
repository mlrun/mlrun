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

import importlib
import os
from typing import Any, Union

import mlrun.common.db.dialects
import mlrun.errors
import mlrun.utils

import framework.utils.db.dsn

_DEFAULT_DRIVER_FOR_DIALECT: dict[str, str] = {
    mlrun.common.db.dialects.Dialects.MYSQL: "pymysql",
    mlrun.common.db.dialects.Dialects.POSTGRESQL: "psycopg2",
    mlrun.common.db.dialects.Dialects.SQLITE: "sqlite3",
}
_ALLOWED_DRIVERS: set[str] = set(_DEFAULT_DRIVER_FOR_DIALECT.values())


class DBUtil:
    _DIALECT = None
    _DSN_ENV = "MLRUN_HTTPDB__DSN"
    _DRIVER_CACHE: dict[str, Any] = {}
    _EMPTY_DB_CONFIGURATIONS = {"nil", "none"}
    _DEFAULT_DB_CONFIGURATIONS = None

    def wait_for_db_liveness(
        self,
        retry_interval: int = 3,
        timeout: int = 120,
    ) -> None:
        """
        Poll the database until a connection succeeds or the timeout is reached.
        """
        mlrun.utils.logger.debug("Waiting for database liveness")
        mlrun.utils.retry_until_successful(
            backoff=retry_interval,
            timeout=timeout,
            logger=mlrun.utils.logger,
            verbose=False,
            _function=self._get_driver().connect,
            **self._connection_kwargs(),
        ).close()
        mlrun.utils.logger.debug("Database is live")

    @classmethod
    def get_dsn(cls) -> str:
        return os.getenv(cls._DSN_ENV, mlrun.mlconf.httpdb.dsn or "")

    def set_configurations(
        self,
        config_items: Union[list[str], dict[str, Any]] | None = None,
    ) -> None:
        items = config_items or self._DEFAULT_DB_CONFIGURATIONS
        keys = _to_keyset(items)

        if not keys or keys.intersection(self._EMPTY_DB_CONFIGURATIONS):
            mlrun.utils.logger.debug(
                "No configurations specified – skipping",
                configs=config_items,
            )
            return

        connection = self._get_connection()
        try:
            self._apply_configurations(connection, items)
        finally:
            connection.close()

    def get_current_configurations(self) -> dict[str, bool]:
        raise NotImplementedError()

    @classmethod
    def get_parsed_dsn(cls) -> framework.utils.db.dsn.Dsn:
        raw_dsn = cls.get_dsn()
        if not raw_dsn:
            raise ValueError("No DSN provided.")

        return framework.utils.db.dsn.Dsn(raw_dsn)

    def _get_connection(self):
        return self._get_driver().connect(**self._connection_kwargs())

    def _connection_kwargs(self) -> dict[str, Any]:
        parsed_dsn = self.get_parsed_dsn()
        settings = {
            "host": parsed_dsn.host,
            "user": parsed_dsn.username,
            "password": parsed_dsn.password,
            "database": parsed_dsn.database,
        }
        if parsed_dsn.port:
            settings["port"] = int(parsed_dsn.port)
        return {key: value for key, value in settings.items() if value is not None}

    def _get_driver(self):
        parser = self.get_parsed_dsn()
        driver_name = parser.driver or _DEFAULT_DRIVER_FOR_DIALECT[parser.dialect]

        if driver_name not in _ALLOWED_DRIVERS:
            raise RuntimeError(
                f"Driver '{driver_name}' is not in the allowed list: {sorted(_ALLOWED_DRIVERS)}"
            )

        if driver_name not in self._DRIVER_CACHE:
            try:
                self._DRIVER_CACHE[driver_name] = importlib.import_module(driver_name)
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"Driver '{driver_name}' required for dialect '{parser.dialect}' "
                    "is not installed.  Install it or adjust the DSN."
                ) from exc

        return self._DRIVER_CACHE[driver_name]

    def __new__(cls, *_, **__) -> "DBUtil":
        if cls is DBUtil:
            dialect = cls.get_parsed_dsn().dialect
            if dialect not in mlrun.common.db.dialects.Dialects.all():
                raise ValueError(
                    f"Unsupported or missing dialect in DSN: {cls.get_dsn()!r}"
                )
            for subclass in cls.__subclasses__():
                if subclass._DIALECT == dialect:
                    return super().__new__(subclass, *_, **__)
            raise RuntimeError(f"No helper registered for dialect {dialect!r}")
        return super().__new__(cls, *_, **__)

    def _apply_configurations(
        self,
        connection: Any,
        config_items: Union[list[str], dict[str, str]],
    ) -> None:
        mlrun.utils.logger.debug("Applying configurations", configs=config_items)


class UtilMySQL(DBUtil):
    _DIALECT = mlrun.common.db.dialects.Dialects.MYSQL
    _DEFAULT_DB_CONFIGURATIONS = mlrun.mlconf.httpdb.db.mysql.modes.split(",")

    def get_current_configurations(self) -> dict[str, bool]:
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT @@GLOBAL.sql_mode;")
                raw = cursor.fetchone()[0] or ""
        except Exception as exc:
            mlrun.utils.logger.exception(
                "Failed to fetch current MySQL configurations",
                error=mlrun.errors.err_to_str(exc),
            )
            raise
        else:
            modes = {
                mode: True for mode in [m.strip() for m in raw.split(",") if m.strip()]
            }
        finally:
            connection.close()
        return modes

    def _apply_configurations(
        self,
        connection: Any,
        config_items: list[str],
    ) -> None:
        modes_csv = ",".join(
            item.strip() for item in config_items if item and item.strip()
        )
        with connection.cursor() as cur:
            cur.execute("SET GLOBAL sql_mode = %s;", (modes_csv,))


class UtilPostgres(DBUtil):
    _DIALECT = mlrun.common.db.dialects.Dialects.POSTGRESQL

    def _connection_kwargs(self) -> dict[str, Any]:
        kw = super()._connection_kwargs()
        kw["dbname"] = kw.pop("database")
        return kw

    def get_current_configurations(self) -> dict[str, str]:
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT name, setting
                    FROM pg_settings
                        """
                )
                modes = {name: value for name, value in cursor.fetchall()}
        except Exception as exc:
            mlrun.utils.logger.exception(
                "Failed to fetch current PostgreSQL configurations",
                error=mlrun.errors.err_to_str(exc),
            )
            raise exc
        else:
            return modes
        finally:
            connection.close()

    def _apply_configurations(
        self,
        connection: Any,
        config_items: Union[list[str], dict[str, str]],
    ) -> None:
        """
        Accepts either a list of "name=value" strings or a dict{name: value},
        validates each GUC, issues ALTER SYSTEM, and reloads.
        """
        if isinstance(config_items, dict):
            setting_pairs = [
                (key.strip(), str(val).strip())
                for key, val in config_items.items()
                if key and str(val).strip()
            ]
        else:
            entries = [e.strip() for e in config_items if e.strip()]
            if not entries:
                mlrun.utils.logger.debug("No valid entries after trimming – skipping")
                return
            try:
                setting_pairs = [
                    (name.strip(), value.strip())
                    for raw in entries
                    for name, value in [raw.split("=", 1)]
                ]
            except ValueError as exc:
                raise ValueError("Each setting must be in 'name=value' format") from exc

        if not setting_pairs:
            mlrun.utils.logger.debug("No valid settings after parsing – skipping")
            return

        # Validate Posgres Grand Unified Configuration names exist
        guc_names = [name for name, _ in setting_pairs]
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT name FROM pg_settings WHERE name = ANY(%s);", (guc_names,)
            )
            existing = {row[0] for row in cursor.fetchall()}
        unknown = [n for n in guc_names if n not in existing]
        if unknown:
            raise ValueError(f"Unknown PostgreSQL GUC(s): {', '.join(unknown)}")

        with connection.cursor() as cursor:
            for param_name, param_value in setting_pairs:
                cursor.execute(f'ALTER SYSTEM SET "{param_name}" = %s;', (param_value,))
            connection.commit()
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_reload_conf();")


class UtilSQLite(DBUtil):
    _DIALECT = mlrun.common.db.dialects.Dialects.SQLITE

    def _connection_kwargs(self) -> dict[str, Any]:
        parsed = self.get_parsed_dsn()
        db_path = parsed._parsed.path.lstrip("/") or ":memory:"
        return {
            "database": db_path,
        }

    def wait_for_db_liveness(self, *_, **__) -> None:  # noqa: D401
        mlrun.utils.logger.debug("SQLite – skipping liveness check")

    def get_current_configurations(self) -> dict[str, str]:
        connection = self._get_connection()
        cursor = connection.cursor()

        cursor.execute("PRAGMA pragma_list;")
        pragma_names = [row[0] for row in cursor.fetchall()]

        configs: dict[str, str] = {}
        for name in pragma_names:
            try:
                cursor.execute(f"PRAGMA {name};")
                val = cursor.fetchone()
                if val is not None:
                    configs[name] = val[0]
            except Exception:
                continue

        cursor.close()
        connection.close()
        return configs

    def _apply_configurations(
        self,
        connection: Any,
        config_items: Union[list[str], dict[str, str]],
    ) -> None:
        if isinstance(config_items, dict):
            items = [f"{k}={v}" for k, v in config_items.items()]
        else:
            items = list(config_items)

        if not items:
            mlrun.utils.logger.debug("No SQLite PRAGMAs to apply – skipping")
            return

        with connection.cursor() as cursor:
            for item in items:
                name, _, value = item.partition("=")
                if not (name and value):
                    raise ValueError(f"Invalid PRAGMA '{item}', expected key=value")
                cursor.execute(f"PRAGMA {name} = {value};")
        connection.commit()


def _to_keyset(
    items: Union[list[str], dict[str, Any]] | None,
) -> set[str] | None:
    if items is None:
        return set()
    if isinstance(items, dict):
        return set(items.keys())
    else:
        return set(items)
