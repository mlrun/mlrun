# Copyright 2026 Iguazio
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

import threading
import time

import sqlalchemy
import sqlalchemy.engine
import sqlalchemy.event
import sqlalchemy.exc

import mlrun.common.db.dialects
import mlrun.common.schemas
import mlrun.config
import mlrun.errors
from mlrun.utils import logger

import framework.db.sqldb.sql_session
import services.api.utils.events.events_factory as events_factory

CATEGORY_DISCONNECT = "disconnect"
CATEGORY_TOO_MANY_CONNECTIONS = "too_many_connections"
CATEGORY_POOL_TIMEOUT = "pool_timeout"
CATEGORY_AUTH_FAILED = "auth_failed"

# Each entry must be unambiguously a "cannot connect" failure. Query-level
# errors (lock waits, deadlocks, query timeouts) stay off the map.
# CATEGORY_AUTH_FAILED covers credential-rotation cases where the API cannot
# open a session at all, e.g. RDS password rotation not yet propagated.
MYSQL_CATEGORIES: dict[int, str] = {
    2002: CATEGORY_DISCONNECT,  # CR_CONNECTION_ERROR (socket)
    2003: CATEGORY_DISCONNECT,  # CR_CONN_HOST_ERROR (TCP)
    2005: CATEGORY_DISCONNECT,  # CR_UNKNOWN_HOST
    2006: CATEGORY_DISCONNECT,  # CR_SERVER_GONE_ERROR
    2013: CATEGORY_DISCONNECT,  # CR_SERVER_LOST
    1040: CATEGORY_TOO_MANY_CONNECTIONS,  # ER_CON_COUNT_ERROR
    1044: CATEGORY_AUTH_FAILED,  # ER_DBACCESS_DENIED_ERROR
    1045: CATEGORY_AUTH_FAILED,  # ER_ACCESS_DENIED_ERROR
    1698: CATEGORY_AUTH_FAILED,  # ER_ACCESS_DENIED_NO_PASSWORD_ERROR
}

# PostgreSQL SQLSTATEs surfaced via psycopg2 ``pgcode`` / psycopg3 ``sqlstate``.
# See https://www.postgresql.org/docs/current/errcodes-appendix.html.
PG_CATEGORIES: dict[str, str] = {
    # Class 08 (connection_exception)
    "08000": CATEGORY_DISCONNECT,
    "08001": CATEGORY_DISCONNECT,  # sqlclient_unable_to_establish_sqlconnection
    "08003": CATEGORY_DISCONNECT,  # connection_does_not_exist
    "08004": CATEGORY_DISCONNECT,  # sqlserver_rejected_establishment_of_sqlconnection
    "08006": CATEGORY_DISCONNECT,  # connection_failure
    # 08007 (transaction_resolution_unknown) is intentionally NOT mapped:
    # it fires when the COMMIT ack is lost, so the txn may have actually
    # committed; reporting it as Connection.Failed would be misleading.
    "57P01": CATEGORY_DISCONNECT,  # admin_shutdown
    "57P02": CATEGORY_DISCONNECT,  # crash_shutdown
    "57P03": CATEGORY_DISCONNECT,  # cannot_connect_now
    "53300": CATEGORY_TOO_MANY_CONNECTIONS,  # too_many_connections
    # Class 28 (invalid_authorization_specification)
    "28000": CATEGORY_AUTH_FAILED,
    "28P01": CATEGORY_AUTH_FAILED,  # invalid_password
}

SUPPORTED_DIALECTS: frozenset[str] = frozenset(
    {
        mlrun.common.db.dialects.Dialects.MYSQL,
        mlrun.common.db.dialects.Dialects.POSTGRESQL,
    }
)

_throttle_lock = threading.Lock()
_last_emit_monotonic: float = 0.0

_registered_engines: set[int] = set()


def classify(
    ctx: sqlalchemy.engine.ExceptionContext,
) -> tuple[str | None, int | str | None]:
    """
    Classify a DBAPI error into ``(category, driver_code)``.

    Returns ``(None, None)`` for errors we don't consider connection-level
    (integrity violations, programming errors, etc.).

    ``driver_code`` is an int for pymysql errors and a SQLSTATE string for
    psycopg2/3 errors.
    """
    original = ctx.original_exception

    if getattr(ctx, "is_disconnect", False):
        # Prefer SQLSTATE: ``args[0]`` may be an OS errno on a wrapped network
        # exception, not a pymysql errno.
        code = _extract_pg_sqlstate(original) or _extract_mysql_code(original)
        return CATEGORY_DISCONNECT, code

    if isinstance(original, sqlalchemy.exc.TimeoutError):
        return CATEGORY_POOL_TIMEOUT, None

    mysql_code = _extract_mysql_code(original)
    if mysql_code is not None:
        category = MYSQL_CATEGORIES.get(mysql_code)
        if category is not None:
            return category, mysql_code

    pg_code = _extract_pg_sqlstate(original)
    if pg_code is not None:
        category = PG_CATEGORIES.get(pg_code)
        if category is not None:
            return category, pg_code

    return None, None


def publish_connection_failed(
    error: BaseException,
    dialect: str | None,
    category: str,
    error_code: int | str | None = None,
) -> bool:
    """
    Best-effort publish of a ``MLRun.DB.Connection.Failed`` event.

    Throttled to one emission per process per
    ``mlconf.events.db_connection.min_emit_interval_seconds``. The throttle
    slot is consumed only on successful delivery; a no-op client or a
    raising ``emit`` leave the slot free so the next DB error can retry.

    :return: True if an event was emitted, False if throttled or unsupported.
    """
    try:
        client = events_factory.EventsFactory.get_events_client()
        event = client.generate_db_connection_event(
            action=mlrun.common.schemas.DBConnectionEventActions.failed,
            error=error,
            error_category=category,
            error_code=error_code,
            dialect=dialect,
        )
        if event is None:
            return False
        previous_slot = _try_claim_emit_slot()
        if previous_slot is None:
            return False
        try:
            client.emit(event)
        except Exception:
            _release_emit_slot(previous_slot)
            raise
        return True
    except Exception as publish_exc:
        logger.warning(
            "Failed to publish DB connection event",
            category=category,
            error_code=error_code,
            exc_info=publish_exc,
        )
        return False


def register(engine: sqlalchemy.engine.Engine) -> None:
    """
    Attach the ``handle_error`` listener to ``engine``. Only MySQL and
    PostgreSQL are wired up; other dialects are skipped (no driver-code
    mapping). The listener never alters exception flow.
    """
    if engine.dialect.name not in SUPPORTED_DIALECTS:
        return
    if id(engine) in _registered_engines:
        return
    _registered_engines.add(id(engine))
    sqlalchemy.event.listen(engine, "handle_error", _on_dbapi_error)


def register_for_default_engine() -> None:
    """
    Attach the connection-failed listener to the runtime MLRun engine.
    Safe to call multiple times (the engine is a process-wide singleton).
    """
    try:
        engine = framework.db.sqldb.sql_session.get_engine()
    except (RuntimeError, AttributeError) as exc:
        logger.warning(
            "Failed to resolve DB engine, skipping connection event listener",
            exc_info=exc,
        )
        return
    register(engine)


def _on_dbapi_error(ctx: sqlalchemy.engine.ExceptionContext) -> None:
    """``handle_error`` listener; never raises."""
    try:
        # ``ctx.engine`` is None during the pool's pre-ping probe, which is
        # self-healing. A real outage surfaces on the fresh-connect attempt
        # that follows, where ``ctx.engine`` is set and we emit.
        if ctx.engine is None:
            return
        category, code = classify(ctx)
        if category is None:
            return
        publish_connection_failed(
            error=ctx.original_exception,
            dialect=ctx.engine.dialect.name,
            category=category,
            error_code=code,
        )
    except Exception as exc:
        logger.warning(
            "Failed to handle DB error event, ignoring",
            exc_info=exc,
        )


def _extract_mysql_code(exc: BaseException | None) -> int | None:
    """
    Return the pymysql errno (``exc.args[0]``), or None if ``exc`` is not a
    pymysql exception. The module check prevents misreporting an unrelated
    int arg (e.g. an OSError errno on a PG disconnect) as a MySQL errno.
    """
    if exc is None:
        return None
    if not type(exc).__module__.startswith(("pymysql", "MySQLdb")):
        return None
    args = exc.args
    if args and isinstance(args[0], int):
        return args[0]
    return None


def _extract_pg_sqlstate(exc: BaseException | None) -> str | None:
    """
    Return the psycopg SQLSTATE (``.pgcode`` for psycopg2, ``.sqlstate`` for
    psycopg3), or None if ``exc`` carries neither.
    """
    if exc is None:
        return None
    code = getattr(exc, "pgcode", None) or getattr(exc, "sqlstate", None)
    if isinstance(code, str) and code:
        return code
    return None


def _try_claim_emit_slot() -> float | None:
    """
    Try to claim the throttle slot. On success returns the previous
    ``_last_emit_monotonic`` value (pass to :func:`_release_emit_slot` to
    undo on delivery failure); returns None if throttled.
    """
    global _last_emit_monotonic
    min_interval = float(mlrun.mlconf.events.db_connection.min_emit_interval_seconds)
    now = time.monotonic()
    with _throttle_lock:
        if now - _last_emit_monotonic < min_interval:
            return None
        previous = _last_emit_monotonic
        _last_emit_monotonic = now
        return previous


def _release_emit_slot(previous: float) -> None:
    """Restore the slot to ``previous`` so the next DB error can retry emit."""
    global _last_emit_monotonic
    with _throttle_lock:
        _last_emit_monotonic = previous
