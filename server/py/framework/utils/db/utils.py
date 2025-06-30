import importlib
import os
from collections.abc import Mapping, Sequence
from typing import Any, Union
from urllib.parse import urlparse

import mlrun.common.db.dialects
import mlrun.utils
from mlrun.errors import err_to_str

_DEFAULT_DRIVER_FOR_DIALECT: dict[str, str] = {
    mlrun.common.db.dialects.Dialects.MYSQL: "pymysql",
    mlrun.common.db.dialects.Dialects.POSTGRESQL: "psycopg2",
}
_ALLOWED_DRIVERS: set[str] = set(_DEFAULT_DRIVER_FOR_DIALECT.values())
_driver_cache: dict[str, Any] = {}


class DBUtil:
    _DIALECT = None
    _DSN_ENV = "MLRUN_HTTPDB__DSN"

    @classmethod
    def _split_scheme(cls, dsn: str) -> tuple[str, str | None]:
        scheme = urlparse(dsn).scheme
        parts = scheme.split("+", 1)
        return parts[0], parts[1] if len(parts) == 2 else None

    def __new__(cls, *_, **__) -> "DBUtil":
        if cls is DBUtil:
            dsn_value = cls.get_dsn()
            dialect, _ = cls._split_scheme(dsn_value)
            if dialect not in mlrun.common.db.dialects.Dialects.all():
                raise ValueError(
                    f"Unsupported or missing dialect in DSN: {dsn_value!r}"
                )
            for subclass in cls.__subclasses__():
                if subclass._DIALECT == dialect:
                    return super().__new__(subclass)
            raise RuntimeError(f"No helper registered for dialect {dialect!r}")
        return super().__new__(cls)

    @classmethod
    def get_dsn(cls) -> str:
        return os.getenv(cls._DSN_ENV, mlrun.mlconf.httpdb.dsn or "")

    def _get_connection(self):
        return self._get_driver().connect(**self._connection_kwargs())

    def _connection_kwargs(self) -> dict[str, Any]:
        parsed = urlparse(self.get_dsn(), allow_fragments=False)
        settings: dict[str, Any] = {
            "host": parsed.hostname,
            "user": parsed.username,
            "password": parsed.password,
            "database": parsed.path.lstrip("/"),
        }
        if parsed.port:
            settings["port"] = parsed.port
        return {k: v for k, v in settings.items() if v is not None}

    def _get_driver(self):
        dialect, explicit_driver = self._split_scheme(self.get_dsn())
        driver_name = explicit_driver or _DEFAULT_DRIVER_FOR_DIALECT[dialect]
        if driver_name not in _ALLOWED_DRIVERS:
            raise RuntimeError(f"Driver '{driver_name}' is not allowed")
        if driver_name not in _driver_cache:
            try:
                _driver_cache[driver_name] = importlib.import_module(driver_name)
            except ModuleNotFoundError as exc:
                raise RuntimeError(f"Driver '{driver_name}' is not installed") from exc
        return _driver_cache[driver_name]

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

    def _apply_configurations(
        self,
        connection: Any,
        config_items: Union[Sequence[str], Mapping[str, str]],
    ) -> None:
        mlrun.utils.logger.debug("Applying configurations", configs=config_items)

    def get_current_configurations(self) -> dict[str, bool]:
        raise NotImplementedError()


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
                "Failed to fetch current MySQL configurations", error=err_to_str(exc)
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
                cur.execute("SELECT name, setting FROM pg_settings;")
                return {name: value for name, value in cur.fetchall()}
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
