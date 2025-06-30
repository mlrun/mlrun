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
from typing import Any, Optional
from urllib.parse import ParseResult, urlparse

import mlrun.utils
import mlrun.common.db.dialects
import mlrun.config

_DEFAULT_DRIVER_FOR_DIALECT: dict[str, str] = {
    mlrun.common.db.dialects.Dialects.MYSQL: "pymysql",
    mlrun.common.db.dialects.Dialects.POSTGRESQL: "psycopg2",
}
_ALLOWED_DRIVERS: set[str] = set(_DEFAULT_DRIVER_FOR_DIALECT.values())
_driver_cache: dict[str, Any] = {}


class DBUtil:
    _DIALECT = None
    _DSN_ENV = "MLRUN_HTTPDB__DSN"

    @staticmethod
    def _split_scheme(dsn: str,) -> tuple[str, Optional[str],]:
        """
        Return (dialect, driver) from the DSN’s scheme.

        'mysql+pymysql://'   → ('mysql', 'pymysql')
        'sqlite:///'         → ('sqlite', None)
        """
        scheme = urlparse(dsn).scheme
        parts = scheme.split("+", 1)
        return parts[0], parts[1] if len(parts) == 2 else None

    def __new__(cls, *_, **__) -> "DBUtil":
        if cls is DBUtil:
            dsn_value = os.getenv(cls._DSN_ENV, mlrun.mlconf.httpdb.dsn or "")
            dialect, _ = cls._split_scheme(dsn_value)
            if dialect not in mlrun.common.db.dialects.Dialects.all():
                raise ValueError(f"Unsupported or missing dialect in DSN: {dsn_value!r}")

            for subclass in cls.__subclasses__():
                if subclass._DIALECT == dialect:
                    return super(DBUtil, subclass).__new__(subclass)

            raise RuntimeError(f"No helper registered for dialect {dialect!r}")
        return super().__new__(cls)

    def _get_connection(self):
        return self._get_driver().connect(**self._connection_kwargs())


    def wait_for_db_liveness(self, retry_interval: int = 3, timeout: int = 120,) -> None:
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

    def set_modes(self, modes: Optional[str]) -> None:
        if not modes or modes.lower() in {"nil", "none"}:
            mlrun.utils.logger.debug("No SQL modes specified – skipping", modes=modes)
            return
        connection = self._get_driver().connect(**self._connection_kwargs())
        try:
            self._apply_modes(connection, modes)
        finally:
            connection.close()

    @classmethod
    def _get_dsn(cls) -> str:
        return os.getenv(cls._DSN_ENV, mlrun.config.config.httpdb.dsn or "")

    def _parse_dsn(self) -> ParseResult:
        return urlparse(self._get_dsn(), allow_fragments=False)

    def _connection_kwargs(self) -> dict[str, Any]:
        parsed = self._parse_dsn()
        settings: dict[str, Any] = {
            "host": parsed.hostname,
            "user": parsed.username,
            "password": parsed.password,
            "database": parsed.path.lstrip("/"),
        }
        if parsed.port:
            settings["port"] = parsed.port
        return {key: value for key, value in settings.items() if value is not None}

    def _get_driver(self):
        dialect, explicit_driver = self._split_scheme(self._get_dsn())
        driver_name = explicit_driver or _DEFAULT_DRIVER_FOR_DIALECT[dialect]

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

    def _apply_modes(self, connection: Any, modes: str,) -> None:
        mlrun.utils.logger.debug("No mode handling for this dialect", modes=modes)


class UtilMySQL(DBUtil):
    _DIALECT = mlrun.common.db.dialects.Dialects.MYSQL

    def _apply_modes(self, connection: Any, modes: str,) -> None:
        with connection.cursor() as cursor:
            cursor.execute("SET GLOBAL sql_mode=%s;", (modes,))

    def _get_current_configurations(self, ) -> set[str]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT @@GLOBAL.sql_mode;")
                guc_strings = (cur.fetchone()[0] or "").strip()
                return {guc.strip() for guc in guc_strings.split(",") if guc.strip()}
        finally:
            conn.close()


class UtilPostgres(DBUtil):
    _DIALECT = mlrun.common.db.dialects.Dialects.POSTGRESQL

    def _connection_kwargs(self) -> dict[str, Any]:
        kwargs = super()._connection_kwargs()
        kwargs["dbname"] = kwargs.pop("database")
        return kwargs

    def _apply_modes(self, connection: Any, modes: str) -> None:
        """
        Accepts a comma-separated list of `guc=value` pairs, validates them,
        stores them via ALTER SYSTEM, then reloads the config.

        Example: 'work_mem=64MB,log_min_duration_statement=1s'
        """
        driver = self._get_driver()
        try:
            sql_mod = driver.sql
        except AttributeError as exc:
            raise RuntimeError(f"Driver {driver.__name__!r} has no 'sql' submodule") from exc

        raw_items = [part.strip() for part in modes.split(",") if part.strip()]
        if not raw_items:
            mlrun.utils.logger.debug("Empty postgres modes string – skipping")
            return

        try:
            guc_pairs = [
                (name.strip(), value.strip())
                for raw in raw_items
                for name, value in [raw.split("=", 1)]
            ]
        except ValueError as exc:
            raise ValueError("Each postgres mode must be in name=value format") from exc

        guc_names = [name for name, _ in guc_pairs]

        # validate GUCs exist
        with connection.cursor() as cur:
            cur.execute(
                "SELECT name FROM pg_settings WHERE name = ANY(%s);",
                (guc_names,),
            )
            existing = {row[0] for row in cur.fetchall()}

        unknown = [n for n in guc_names if n not in existing]
        if unknown:
            raise ValueError(f"Unknown PostgreSQL GUC(s): {', '.join(unknown)}")

        with connection.cursor() as cur:
            for name, value in guc_pairs:
                stmt = sql_mod.SQL("ALTER SYSTEM SET {} = %s").format(
                    sql_mod.Identifier(name)
                )
                cur.execute(stmt, (value,))
            cur.execute("SELECT pg_reload_conf();")
