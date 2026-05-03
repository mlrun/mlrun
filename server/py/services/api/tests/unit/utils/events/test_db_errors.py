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
import unittest.mock

import pymysql.err
import pytest
import sqlalchemy
import sqlalchemy.exc

import mlrun
import mlrun.common.schemas

import framework.db.sqldb.sql_session
import services.api.utils.events.db_errors as db_errors


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(db_errors, "_last_emit_monotonic", 0.0)
    db_errors._registered_engines.clear()
    yield
    db_errors._registered_engines.clear()


@pytest.fixture
def mysql_engine(monkeypatch):
    """In-memory engine masquerading as MySQL so register() attaches a listener."""
    return _engine_with_dialect(monkeypatch, "mysql")


@pytest.mark.parametrize(
    "exc,is_disconnect,expected_category,expected_code",
    [
        (
            pymysql.err.OperationalError(2013, "Lost connection to MySQL server"),
            True,
            db_errors.CATEGORY_DISCONNECT,
            2013,
        ),
        (
            pymysql.err.OperationalError(2003, "Can't connect to MySQL server"),
            False,
            db_errors.CATEGORY_DISCONNECT,
            2003,
        ),
        (
            pymysql.err.OperationalError(1040, "Too many connections"),
            False,
            db_errors.CATEGORY_TOO_MANY_CONNECTIONS,
            1040,
        ),
        (
            pymysql.err.OperationalError(1045, "Access denied for user"),
            False,
            db_errors.CATEGORY_AUTH_FAILED,
            1045,
        ),
        (
            pymysql.err.OperationalError(1044, "Access denied for user to db"),
            False,
            db_errors.CATEGORY_AUTH_FAILED,
            1044,
        ),
        (
            pymysql.err.OperationalError(1698, "Access denied (no password)"),
            False,
            db_errors.CATEGORY_AUTH_FAILED,
            1698,
        ),
    ],
)
def test_classify_mysql_codes(exc, is_disconnect, expected_category, expected_code):
    category, code = db_errors.classify(_ctx(exc, is_disconnect=is_disconnect))
    assert category == expected_category
    assert code == expected_code


@pytest.mark.parametrize(
    "exc",
    [
        # Lock waits, deadlocks, and query timeouts are query-level. They
        # must NOT trigger a connection-failed event.
        pymysql.err.OperationalError(1205, "Lock wait timeout exceeded"),
        pymysql.err.OperationalError(1213, "Deadlock found"),
        pymysql.err.OperationalError(3024, "Query execution was interrupted"),
    ],
)
def test_classify_mysql_query_level_errors_skipped(exc):
    """Conservative mapping: query-level errors don't fire Connection.Failed."""
    category, code = db_errors.classify(_ctx(exc))
    assert category is None
    assert code is None


def test_classify_pool_timeout():
    pool_timeout = sqlalchemy.exc.TimeoutError("QueuePool checkout timed out")
    category, code = db_errors.classify(_ctx(pool_timeout))
    assert category == db_errors.CATEGORY_POOL_TIMEOUT
    assert code is None


class _PsycopgLikeError(Exception):
    """Stand-in for psycopg2/3 errors: carries a SQLSTATE on .pgcode/.sqlstate."""

    def __init__(self, sqlstate: str, msg: str = ""):
        super().__init__(msg or sqlstate)
        self.pgcode = sqlstate
        self.sqlstate = sqlstate


@pytest.mark.parametrize(
    "sqlstate,expected_category",
    [
        ("08006", db_errors.CATEGORY_DISCONNECT),  # connection_failure
        ("08001", db_errors.CATEGORY_DISCONNECT),
        ("57P03", db_errors.CATEGORY_DISCONNECT),  # cannot_connect_now
        ("57P01", db_errors.CATEGORY_DISCONNECT),  # admin_shutdown
        ("53300", db_errors.CATEGORY_TOO_MANY_CONNECTIONS),
        (
            "28000",
            db_errors.CATEGORY_AUTH_FAILED,
        ),  # invalid_authorization_specification
        ("28P01", db_errors.CATEGORY_AUTH_FAILED),  # invalid_password
    ],
)
def test_classify_postgres_sqlstates(sqlstate, expected_category):
    exc = _PsycopgLikeError(sqlstate)
    category, code = db_errors.classify(_ctx(exc, dialect_name="postgresql"))
    assert category == expected_category
    assert code == sqlstate


@pytest.mark.parametrize(
    "sqlstate",
    [
        "55P03",  # lock_not_available
        "40P01",  # deadlock_detected
        "57014",  # query_canceled (statement_timeout)
    ],
)
def test_classify_postgres_query_level_sqlstates_skipped(sqlstate):
    """Conservative mapping: query-level SQLSTATEs don't fire Connection.Failed."""
    exc = _PsycopgLikeError(sqlstate)
    category, code = db_errors.classify(_ctx(exc, dialect_name="postgresql"))
    assert category is None
    assert code is None


def test_classify_postgres_disconnect_via_is_disconnect_flag():
    """SQLAlchemy may set is_disconnect=True without an obvious SQLSTATE."""
    exc = _PsycopgLikeError("08006", "server closed the connection unexpectedly")
    category, code = db_errors.classify(
        _ctx(exc, is_disconnect=True, dialect_name="postgresql")
    )
    assert category == db_errors.CATEGORY_DISCONNECT
    assert code == "08006"


def test_classify_postgres_disconnect_prefers_sqlstate_over_stray_int_arg():
    """
    On a PG disconnect, ``args[0]`` may be an OS errno (e.g. 110 from a
    network-level OSError wrapped by SQLAlchemy). The reported error_code
    must be the SQLSTATE; the stray int must not leak through.
    """

    class _MixedError(Exception):
        def __init__(self):
            super().__init__(110, "Connection timed out")
            self.pgcode = "08006"
            self.sqlstate = "08006"

    exc = _MixedError()
    category, code = db_errors.classify(
        _ctx(exc, is_disconnect=True, dialect_name="postgresql")
    )
    assert category == db_errors.CATEGORY_DISCONNECT
    assert code == "08006"


def test_classify_postgres_unknown_sqlstate_returns_none():
    """Non-fatal SQLSTATE (e.g. 23505 unique violation) must not trigger event."""
    exc = _PsycopgLikeError("23505", "unique_violation")
    category, code = db_errors.classify(_ctx(exc, dialect_name="postgresql"))
    assert category is None
    assert code is None


@pytest.mark.parametrize(
    "exc",
    [
        pymysql.err.IntegrityError(1062, "Duplicate entry"),
        pymysql.err.ProgrammingError(1064, "SQL syntax error"),
        ValueError("not a db error"),
    ],
)
def test_classify_skips_non_fatal_errors(exc):
    category, code = db_errors.classify(_ctx(exc))
    assert category is None
    assert code is None


def test_publish_emits_event_via_factory(monkeypatch):
    fake_event = object()
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_db_connection_event.return_value = fake_event
    monkeypatch.setattr(
        db_errors.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(return_value=fake_client),
    )

    err = pymysql.err.OperationalError(2013, "Lost connection to MySQL server")
    emitted = db_errors.publish_connection_failed(
        error=err,
        dialect="mysql",
        category=db_errors.CATEGORY_DISCONNECT,
        error_code=2013,
    )
    assert emitted is True
    fake_client.generate_db_connection_event.assert_called_once()
    call_kwargs = fake_client.generate_db_connection_event.call_args.kwargs
    assert call_kwargs["action"] == mlrun.common.schemas.DBConnectionEventActions.failed
    assert call_kwargs["error"] is err
    assert call_kwargs["error_category"] == db_errors.CATEGORY_DISCONNECT
    assert call_kwargs["error_code"] == 2013
    assert call_kwargs["dialect"] == "mysql"
    fake_client.emit.assert_called_once_with(fake_event)


def test_publish_no_event_from_nop_client_does_not_consume_throttle(monkeypatch):
    """A NopClient returns None: emit is skipped AND the throttle slot stays free."""
    nop_client = unittest.mock.MagicMock()
    nop_client.generate_db_connection_event.return_value = None
    real_client = unittest.mock.MagicMock()
    real_client.generate_db_connection_event.return_value = object()

    # First call returns nop, second call returns a real client; if the nop call
    # had consumed the slot, the second emit would be throttled.
    monkeypatch.setattr(
        db_errors.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(side_effect=[nop_client, real_client]),
    )

    assert (
        db_errors.publish_connection_failed(
            RuntimeError("boom"), "mysql", db_errors.CATEGORY_DISCONNECT
        )
        is False
    )
    nop_client.emit.assert_not_called()

    assert (
        db_errors.publish_connection_failed(
            RuntimeError("boom"), "mysql", db_errors.CATEGORY_DISCONNECT
        )
        is True
    )
    real_client.emit.assert_called_once()


def test_publish_releases_slot_when_emit_raises(monkeypatch):
    """
    If ``client.emit`` raises (e.g. events service unreachable), no event was
    delivered, so the throttle slot must be released so the next DB error
    within the window can retry.
    """
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_db_connection_event.return_value = object()
    fake_client.emit.side_effect = [
        RuntimeError("events service unreachable"),
        None,
    ]
    monkeypatch.setattr(
        db_errors.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(return_value=fake_client),
    )
    monkeypatch.setattr(
        mlrun.mlconf.events.db_connection, "min_emit_interval_seconds", 60
    )
    fake_now = {"value": 1000.0}
    monkeypatch.setattr(db_errors.time, "monotonic", lambda: fake_now["value"])

    err = pymysql.err.OperationalError(2013, "Lost connection")
    # First emit fails; slot must be released so we are not locked out.
    assert (
        db_errors.publish_connection_failed(err, "mysql", db_errors.CATEGORY_DISCONNECT)
        is False
    )
    # Same instant: a slot was released, so the next attempt can claim & emit.
    assert (
        db_errors.publish_connection_failed(err, "mysql", db_errors.CATEGORY_DISCONNECT)
        is True
    )
    assert fake_client.emit.call_count == 2


def test_publish_keeps_slot_after_success(monkeypatch):
    """A successful emit must keep the slot consumed (otherwise no throttle)."""
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_db_connection_event.return_value = object()
    monkeypatch.setattr(
        db_errors.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(return_value=fake_client),
    )
    monkeypatch.setattr(
        mlrun.mlconf.events.db_connection, "min_emit_interval_seconds", 60
    )
    fake_now = {"value": 1000.0}
    monkeypatch.setattr(db_errors.time, "monotonic", lambda: fake_now["value"])

    err = pymysql.err.OperationalError(2013, "Lost connection")
    assert (
        db_errors.publish_connection_failed(err, "mysql", db_errors.CATEGORY_DISCONNECT)
        is True
    )
    fake_now["value"] += 30
    assert (
        db_errors.publish_connection_failed(err, "mysql", db_errors.CATEGORY_DISCONNECT)
        is False
    )
    assert fake_client.emit.call_count == 1


def test_publish_swallows_factory_exception(monkeypatch):
    monkeypatch.setattr(
        db_errors.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(side_effect=RuntimeError("network down")),
    )
    emitted = db_errors.publish_connection_failed(
        error=RuntimeError("boom"),
        dialect="mysql",
        category=db_errors.CATEGORY_DISCONNECT,
    )
    assert emitted is False


def test_publish_throttles_within_interval(monkeypatch):
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_db_connection_event.return_value = object()
    monkeypatch.setattr(
        db_errors.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(return_value=fake_client),
    )
    monkeypatch.setattr(
        mlrun.mlconf.events.db_connection, "min_emit_interval_seconds", 60
    )
    fake_now = {"value": 1000.0}
    monkeypatch.setattr(db_errors.time, "monotonic", lambda: fake_now["value"])

    err = pymysql.err.OperationalError(2013, "Lost connection")
    assert (
        db_errors.publish_connection_failed(err, "mysql", db_errors.CATEGORY_DISCONNECT)
        is True
    )

    fake_now["value"] += 30
    assert (
        db_errors.publish_connection_failed(err, "mysql", db_errors.CATEGORY_DISCONNECT)
        is False
    )

    fake_now["value"] += 60
    assert (
        db_errors.publish_connection_failed(err, "mysql", db_errors.CATEGORY_DISCONNECT)
        is True
    )
    assert fake_client.emit.call_count == 2


def test_publish_throttle_interval_is_configurable(monkeypatch):
    fake_client = unittest.mock.MagicMock()
    fake_client.generate_db_connection_event.return_value = object()
    monkeypatch.setattr(
        db_errors.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(return_value=fake_client),
    )
    # Tighten the throttle to 5 seconds
    monkeypatch.setattr(
        mlrun.mlconf.events.db_connection, "min_emit_interval_seconds", 5
    )
    fake_now = {"value": 1000.0}
    monkeypatch.setattr(db_errors.time, "monotonic", lambda: fake_now["value"])

    err = pymysql.err.OperationalError(2013, "Lost connection")
    assert (
        db_errors.publish_connection_failed(err, "mysql", db_errors.CATEGORY_DISCONNECT)
        is True
    )
    fake_now["value"] += 6  # past the 5s window
    assert (
        db_errors.publish_connection_failed(err, "mysql", db_errors.CATEGORY_DISCONNECT)
        is True
    )
    assert fake_client.emit.call_count == 2


@pytest.mark.parametrize(
    "dialect,expected_attached",
    [
        ("mysql", True),
        ("postgresql", True),
        ("sqlite", False),
    ],
)
def test_register_attaches_listener_per_dialect(
    monkeypatch, dialect, expected_attached
):
    """Listener attaches for supported dialects (mysql, postgresql) and skips others."""
    engine = _engine_with_dialect(monkeypatch, dialect)
    db_errors.register(engine)
    assert (
        sqlalchemy.event.contains(engine, "handle_error", db_errors._on_dbapi_error)
        is expected_attached
    )
    assert (id(engine) in db_errors._registered_engines) is expected_attached


def test_register_is_idempotent_per_engine(mysql_engine, monkeypatch):
    listen_spy = unittest.mock.MagicMock(wraps=sqlalchemy.event.listen)
    monkeypatch.setattr(sqlalchemy.event, "listen", listen_spy)

    db_errors.register(mysql_engine)
    db_errors.register(mysql_engine)
    assert listen_spy.call_count == 1


def test_listener_publishes_on_fatal_error(monkeypatch):
    publish_calls: list[dict] = []
    monkeypatch.setattr(
        db_errors,
        "publish_connection_failed",
        lambda **kw: publish_calls.append(kw) or True,
    )
    ctx = _ctx(pymysql.err.OperationalError(2013, "Lost connection"))
    db_errors._on_dbapi_error(ctx)

    assert len(publish_calls) == 1
    assert publish_calls[0]["category"] == db_errors.CATEGORY_DISCONNECT
    assert publish_calls[0]["error_code"] == 2013
    assert publish_calls[0]["dialect"] == "mysql"


def test_listener_skips_non_fatal_error(monkeypatch):
    publish_calls: list = []
    monkeypatch.setattr(
        db_errors,
        "publish_connection_failed",
        lambda **kw: publish_calls.append(kw) or True,
    )
    ctx = _ctx(pymysql.err.IntegrityError(1062, "Duplicate"))
    db_errors._on_dbapi_error(ctx)
    assert publish_calls == []


def test_listener_skips_pool_pre_ping(monkeypatch):
    """Pre-ping (``ctx.engine is None``) must not emit an event."""
    publish_calls: list = []
    monkeypatch.setattr(
        db_errors,
        "publish_connection_failed",
        lambda **kw: publish_calls.append(kw) or True,
    )
    classify_spy = unittest.mock.MagicMock()
    monkeypatch.setattr(db_errors, "classify", classify_spy)

    ctx = _ctx(
        pymysql.err.OperationalError(2013, "Lost connection during pre-ping"),
        is_disconnect=True,
    )
    ctx.engine = None

    db_errors._on_dbapi_error(ctx)

    assert publish_calls == []
    classify_spy.assert_not_called()


def test_listener_swallows_internal_errors(monkeypatch):
    """If classify or publish raises, the listener must not propagate."""
    monkeypatch.setattr(
        db_errors,
        "classify",
        unittest.mock.MagicMock(side_effect=RuntimeError("classifier broke")),
    )
    ctx = _ctx(pymysql.err.OperationalError(2013, "lost"), is_disconnect=True)
    # Must not raise
    db_errors._on_dbapi_error(ctx)


def test_register_for_default_engine_uses_session_engine(monkeypatch):
    fake_engine = unittest.mock.MagicMock()
    fake_engine.dialect.name = "mysql"
    monkeypatch.setattr(
        framework.db.sqldb.sql_session, "get_engine", lambda: fake_engine
    )
    listen_spy = unittest.mock.MagicMock()
    monkeypatch.setattr(sqlalchemy.event, "listen", listen_spy)

    db_errors.register_for_default_engine()

    listen_spy.assert_called_once()
    args = listen_spy.call_args.args
    assert args[0] is fake_engine
    assert args[1] == "handle_error"


def test_register_for_default_engine_swallows_engine_lookup_failure(monkeypatch):
    monkeypatch.setattr(
        framework.db.sqldb.sql_session,
        "get_engine",
        unittest.mock.MagicMock(side_effect=RuntimeError("no DSN")),
    )
    # Must not raise
    db_errors.register_for_default_engine()


def _ctx(exception, is_disconnect=False, dialect_name="mysql"):
    """Build a minimal stand-in for sqlalchemy.engine.ExceptionContext.

    ``ExceptionContext.engine`` is None during pool pre-ping (the listener uses
    that as a signal to skip emitting the event); set ``ctx.engine = None``
    explicitly on a returned ctx to simulate that path.
    """
    ctx = unittest.mock.MagicMock()
    ctx.original_exception = exception
    ctx.is_disconnect = is_disconnect
    ctx.engine.dialect.name = dialect_name
    ctx.dialect.name = dialect_name
    return ctx


def _engine_with_dialect(
    monkeypatch: pytest.MonkeyPatch, dialect_name: str
) -> sqlalchemy.engine.Engine:
    """In-memory SQLite engine with its dialect name patched to ``dialect_name``."""
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    monkeypatch.setattr(engine.dialect, "name", dialect_name)
    return engine
