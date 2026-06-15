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
"""
Integration tests for ``MLRun.DB.Connection.Failed`` against a real
MySQL or PostgreSQL container. Selected via ``MLRUN_TEST_DB={mysql,postgres}``.

The mapping is intentionally conservative: only true "cannot connect"
failures (driver-level disconnects, ``too_many_connections``, pool checkout
timeout) emit the event. Everything else, including lock waits, deadlocks,
query timeouts, and integrity violations, must NOT trigger it.

These tests exercise the wiring against a real driver and verify the
no-false-positive contract for the most plausible noise sources:
  * Lock wait timeout: common under contention; must NOT emit.
  * Integrity violation: user error; must NOT emit.

True-positive disconnect detection is covered by the unit tests
(test_db_errors.py) and was verified end-to-end on the lab during PR review.
Reproducing a real disconnect deterministically in a container fixture is
brittle (it usually means killing the container mid-test).
"""

import threading
import unittest.mock

import pytest
import sqlalchemy
import sqlalchemy.exc

import mlrun.common.db.dialects
import mlrun.common.schemas

import services.api.utils.events.db_errors as db_errors


@pytest.fixture
def stub_events_client(monkeypatch):
    """Replace the events factory with a stub that records emitted specs."""
    emitted: list = []
    spec_marker = object()

    client = unittest.mock.MagicMock()
    client.generate_db_connection_event.return_value = spec_marker
    client.emit.side_effect = lambda spec: emitted.append(
        client.generate_db_connection_event.call_args.kwargs
    )

    monkeypatch.setattr(
        db_errors.events_factory.EventsFactory,
        "get_events_client",
        unittest.mock.MagicMock(return_value=client),
    )
    monkeypatch.setattr(db_errors._slot, "_last_emit_monotonic", 0.0)
    db_errors._registered_engines.clear()
    yield emitted
    db_errors._registered_engines.clear()


@pytest.fixture
def registered_engine(db_engine, stub_events_client):
    db_errors.register(db_engine)
    return db_engine


def test_lock_wait_timeout_does_not_emit_event(
    registered_engine: sqlalchemy.engine.Engine,
    stub_events_client: list,
) -> None:
    """
    Lock wait timeouts are NOT a connection failure: the connection is fine,
    another transaction held a row lock too long. Must NOT emit the event.
    """
    if _is_mysql(registered_engine):
        sql_lock_timeout_setup = "SET SESSION innodb_lock_wait_timeout = 1"
    elif _is_postgres(registered_engine):
        sql_lock_timeout_setup = "SET lock_timeout = '500ms'"
    else:
        pytest.skip("Unsupported dialect")

    with registered_engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                "CREATE TABLE conn_event_lock_test (id INT PRIMARY KEY, v INT)"
            )
        )
        conn.execute(sqlalchemy.text("INSERT INTO conn_event_lock_test VALUES (1, 1)"))

    blocker_holding = threading.Event()
    release_blocker = threading.Event()

    def _blocker():
        with registered_engine.begin() as bconn:
            bconn.execute(
                sqlalchemy.text(
                    "SELECT * FROM conn_event_lock_test WHERE id = 1 FOR UPDATE"
                )
            )
            blocker_holding.set()
            release_blocker.wait(timeout=10)

    blocker_thread = threading.Thread(target=_blocker)
    blocker_thread.start()
    try:
        assert blocker_holding.wait(timeout=10), "blocker never acquired the lock"

        with pytest.raises(sqlalchemy.exc.DBAPIError):
            with registered_engine.connect() as conn:
                conn.execute(sqlalchemy.text(sql_lock_timeout_setup))
                conn.execute(sqlalchemy.text("BEGIN"))
                conn.execute(
                    sqlalchemy.text(
                        "SELECT * FROM conn_event_lock_test WHERE id = 1 FOR UPDATE"
                    )
                )
    finally:
        release_blocker.set()
        blocker_thread.join(timeout=10)
        with registered_engine.begin() as conn:
            conn.execute(sqlalchemy.text("DROP TABLE conn_event_lock_test"))

    assert stub_events_client == [], (
        "lock wait timeout must not trigger Connection.Failed (false-positive risk)"
    )


def test_integrity_violation_does_not_emit_event(
    registered_engine: sqlalchemy.engine.Engine,
    stub_events_client: list,
) -> None:
    """A duplicate-key integrity error is a user error, not a connection failure."""
    with registered_engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                "CREATE TABLE conn_event_integrity_test (id INT PRIMARY KEY)"
            )
        )
        conn.execute(
            sqlalchemy.text("INSERT INTO conn_event_integrity_test VALUES (1)")
        )

    try:
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            with registered_engine.begin() as conn:
                conn.execute(
                    sqlalchemy.text("INSERT INTO conn_event_integrity_test VALUES (1)")
                )
    finally:
        with registered_engine.begin() as conn:
            conn.execute(sqlalchemy.text("DROP TABLE conn_event_integrity_test"))

    assert stub_events_client == [], "integrity violation must not trigger an event"


def _is_postgres(engine: sqlalchemy.engine.Engine) -> bool:
    return engine.dialect.name.startswith(mlrun.common.db.dialects.Dialects.POSTGRESQL)


def _is_mysql(engine: sqlalchemy.engine.Engine) -> bool:
    return engine.dialect.name.startswith(mlrun.common.db.dialects.Dialects.MYSQL)
