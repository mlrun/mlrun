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
import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union
from urllib.parse import parse_qs, urlparse

import mlrun.common.db.dialects
import mlrun.errors
import mlrun.utils

_DEFAULT_DRIVER_FOR_DIALECT: dict[str, str] = {
    mlrun.common.db.dialects.Dialects.MYSQL: "pymysql",
    mlrun.common.db.dialects.Dialects.POSTGRESQL: "psycopg2",
}
_ALLOWED_DRIVERS: set[str] = set(_DEFAULT_DRIVER_FOR_DIALECT.values())
_driver_cache: dict[str, Any] = {}

from typing import Any

import mlrun.common.db.dialects


class ParsedDsn:
    _IDENTIFIER_REGEX = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")
    _PATH_REGEX = re.compile(r"[A-Za-z0-9_\-\.\/]+")

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._parsed = urlparse(dsn)
        self.dialect, self.driver = self._split_scheme(
            scheme=self._parsed.scheme,
        )

        # Connection components
        if self.dialect == mlrun.common.db.dialects.Dialects.SQLITE:
            self.username = None
            self.password = None
            self.host = None
            self.port = None
            # SQLite DSNs ignore database path
            self.database = None
        else:
            self.username = self._parsed.username
            self.password = self._parsed.password
            self.host = self._parsed.hostname
            self.port = self._parsed.port
            self.database = self._parsed.path.lstrip("/") or None

        # Query configurations
        if self._parsed.query:
            raw_qs = parse_qs(self._parsed.query)
            self.configurations: dict[str, Union[str, list[str]]] = {
                key: (value[0] if len(value) == 1 else value)
                for key, value in raw_qs.items()
            }
        else:
            self.configurations = {}

    def is_valid(self) -> bool:
        """
        Validates the DSN by:
        1. Ensuring dialect is supported
        2. Checking driver identifier format
        3. For SQLite: always valid here
        4. For others: validating database path/name and other components
        """
        if (
            not self.dialect
            or self.dialect not in mlrun.common.db.dialects.Dialects.all()
        ):
            return False
        if self.driver and not self._IDENTIFIER_REGEX.fullmatch(self.driver):
            return False
        raw_path = self._parsed.path.lstrip("/")
        if self.dialect == mlrun.common.db.dialects.Dialects.SQLITE:
            # SQLite DSNs: validate optional file path or :memory:
            if not raw_path or raw_path == ":memory:":
                return True
        else:
            if not self._PATH_REGEX.fullmatch(raw_path):
                return False
        if self.dialect == mlrun.common.db.dialects.Dialects.SQLITE:
            return True
        # Database validation for non-SQLite
        if not (self.database and self._PATH_REGEX.fullmatch(self.database)):
            return False
        # Non-SQLite: username, host, optional port
        if not self.username:
            return False
        if not (self.host and self._IDENTIFIER_REGEX.fullmatch(self.host)):
            return False
        if self.port is not None and not (1 <= self.port <= 65535):
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "dialect": self.dialect,
            "driver": self.driver,
            "username": self.username,
            "password": self.password,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "configurations": self.configurations,
        }

    @staticmethod
    def _split_scheme(scheme: str) -> tuple[str, Optional[str]]:
        parts = scheme.split("+", 1)
        return parts[0], parts[1] if len(parts) == 2 else None


class DBUtil:
    _DIALECT = None
    _DSN_ENV = "MLRUN_HTTPDB__DSN"

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
            retry_interval,
            timeout,
            mlrun.utils.logger,
            raise_on_failure=True,
            func=self._get_driver().connect,
            **self._connection_kwargs(),
        ).close()
        mlrun.utils.logger.debug("Database is live")

    @classmethod
    def get_dsn(cls) -> str:
        return os.getenv(cls._DSN_ENV, mlrun.mlconf.httpdb.dsn or "")

    def set_configurations(
        self,
        config_items: Union[list[str], dict[str, str]],
    ) -> None:
        if not config_items:
            mlrun.utils.logger.debug(
                "No configurations specified – skipping", configs=config_items
            )
            return
        conn = self._get_driver().connect(**self._connection_kwargs())
        try:
            self._apply_configurations(conn, config_items)
        finally:
            conn.close()

    def get_current_configurations(self) -> dict[str, bool]:
        raise NotImplementedError()

    @classmethod
    def get_parsed_dsn(cls) -> ParsedDsn:
        return ParsedDsn(cls.get_dsn())

    def _get_connection(self):
        return self._get_driver().connect(**self._connection_kwargs())

    def _connection_kwargs(self) -> dict[str, Any]:
        parsed_dsn = ParsedDsn(self.get_dsn())
        settings = {
            "host": parsed_dsn.host,
            "user": parsed_dsn.username,
            "password": parsed_dsn.password,
            "database": parsed_dsn.database,
        }
        if parsed_dsn.port:
            settings["port"] = str(parsed_dsn.port)
        return {key: value for key, value in settings.items() if value is not None}

    def _get_driver(self):
        parser = ParsedDsn(self.get_dsn())
        driver_name = parser.driver or _DEFAULT_DRIVER_FOR_DIALECT[parser.dialect]

        if driver_name not in _ALLOWED_DRIVERS:
            raise RuntimeError(
                f"Driver '{driver_name}' is not in the allowed list: {sorted(_ALLOWED_DRIVERS)}"
            )

        if driver_name not in _driver_cache:
            try:
                _driver_cache[driver_name] = importlib.import_module(driver_name)
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"Driver '{driver_name}' required for dialect '{dialect}' "
                    "is not installed.  Install it or adjust the DSN."
                ) from exc

        return _driver_cache[driver_name]

    @classmethod
    def _split_scheme(cls, dsn: str) -> tuple[str, Optional[str]]:
        scheme = urlparse(dsn).scheme
        parts = scheme.split("+", 1)
        return parts[0], parts[1] if len(parts) == 2 else None

    def __new__(cls, *_, **__) -> "DBUtil":
        if cls is DBUtil:
            dsn_value = cls.get_dsn()
            dialect = ParsedDsn(dsn_value).dialect
            if dialect not in mlrun.common.db.dialects.Dialects.all():
                raise ValueError(
                    f"Unsupported or missing dialect in DSN: {dsn_value!r}"
                )
            for subclass in cls.__subclasses__():
                if subclass._DIALECT == dialect:
                    return super().__new__(subclass)
            raise RuntimeError(f"No helper registered for dialect {dialect!r}")
        return super().__new__(cls)

    def _apply_configurations(
        self,
        connection: Any,
        config_items: Union[Sequence[str], Mapping[str, str]],
    ) -> None:
        mlrun.utils.logger.debug("Applying configurations", configs=config_items)


class UtilMySQL(DBUtil):
    _DIALECT = mlrun.common.db.dialects.Dialects.MYSQL

    def get_current_configurations(self) -> dict[str, bool]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT @@GLOBAL.sql_mode;")
                raw = cur.fetchone()[0] or ""
                modes = [m.strip() for m in raw.split(",") if m.strip()]
                return {mode: True for mode in modes}
        except Exception as exc:
            mlrun.utils.logger.exception(
                "Failed to fetch current MySQL configurations",
                error=mlrun.errors.err_to_str(exc),
            )
            raise
        finally:
            conn.close()

    def _apply_configurations(
        self,
        connection: Any,
        config_items: Sequence[str],
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
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT name, setting
                    FROM pg_settings
                        """
                )
                return {name: value for name, value in cur.fetchall()}
        except Exception as exc:
            mlrun.utils.logger.exception(
                "Failed to fetch current PostgreSQL configurations",
                error=mlrun.errors.err_to_str(exc),
            )
            raise exc
        finally:
            conn.close()

    def _apply_configurations(
        self,
        connection: Any,
        config_items: Union[list[str], dict[str, str]],
    ) -> None:
        """
        Accepts either a list of "name=value" strings or a dict{name: value},
        validates each GUC, issues ALTER SYSTEM, and reloads.
        """
        if isinstance(config_items, Mapping):
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

        # Validate GUC names exist
        guc_names = [name for name, _ in setting_pairs]
        connection.autocommit = True
        with connection.cursor() as cur:
            cur.execute(
                "SELECT name FROM pg_settings WHERE name = ANY(%s);", (guc_names,)
            )
            existing = {row[0] for row in cur.fetchall()}
        unknown = [n for n in guc_names if n not in existing]
        if unknown:
            raise ValueError(f"Unknown PostgreSQL GUC(s): {', '.join(unknown)}")

        with connection.cursor() as cur:
            for param_name, param_value in setting_pairs:
                cur.execute(f'ALTER SYSTEM SET "{param_name}" = %s;', (param_value,))
            connection.commit()
        with connection.cursor() as cur:
            cur.execute("SELECT pg_reload_conf();")
