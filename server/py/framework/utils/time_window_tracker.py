# Copyright 2023 Iguazio
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

import asyncio
import datetime
import typing

import sqlalchemy.orm

import mlrun.common.types

import framework.db.session
import framework.utils.asyncio
import framework.utils.singletons.db


class TimeWindowTrackerKeys(mlrun.common.types.StrEnum):
    run_monitoring = "run_monitoring"
    log_collection = "log_collection"
    events_generation = "events_generation"


class TimeWindowTracker:
    def __init__(
        self,
        key: str,
        max_window_size_seconds: int | None = None,
    ):
        self._key = key
        self._timestamp = None
        self._max_window_size_seconds = max_window_size_seconds

        self._db = framework.utils.singletons.db.get_db()

    def initialize(self, session: sqlalchemy.orm.Session):
        time_window_tracker_record = self._refresh_from_db(
            session, raise_on_not_found=False
        )
        self._timestamp = self._timestamp or datetime.datetime.now(datetime.UTC)
        if not time_window_tracker_record:
            self._db.store_time_window_tracker_record(
                session, self._key, self._timestamp, self._max_window_size_seconds
            )

    def update_window(
        self,
        session: sqlalchemy.orm.Session,
        timestamp: datetime.datetime | None = None,
    ):
        self._timestamp = timestamp or datetime.datetime.now(datetime.UTC)
        self._db.store_time_window_tracker_record(
            session, self._key, self._timestamp, self._max_window_size_seconds
        )

    def get_window(self, session: sqlalchemy.orm.Session) -> datetime.datetime:
        self._refresh_from_db(session, raise_on_not_found=True)
        return self._timestamp

    def _refresh_from_db(
        self, session: sqlalchemy.orm.Session, raise_on_not_found: bool = True
    ):
        time_window_tracker_record = self._db.get_time_window_tracker_record(
            session,
            self._key,
            raise_on_not_found=raise_on_not_found,
        )
        if not time_window_tracker_record:
            return

        # Ensure the timestamp is timezone-aware, it might return as naive from the DB
        # though it was saved as timezone-aware
        self._timestamp = time_window_tracker_record.timestamp.replace(
            tzinfo=datetime.UTC
        )
        self._max_window_size_seconds = (
            time_window_tracker_record.max_window_size_seconds
        )
        if time_window_tracker_record.max_window_size_seconds is not None:
            self._timestamp = max(
                self._timestamp,
                datetime.datetime.now(datetime.UTC)
                - datetime.timedelta(seconds=self._max_window_size_seconds),
            )
            self.update_window(session, self._timestamp)

        return time_window_tracker_record


async def run_with_time_window_tracker(
    key: TimeWindowTrackerKeys,
    max_window_size_seconds: int,
    ensure_window_update: bool,
    callback: typing.Callable,
    *args,
    **kwargs,
):
    """
    Runs the given callback within a time window tracked by TimeWindowTracker.
    Use this function when you are in an async context and your callback is async.
    """
    cycle_tracker = TimeWindowTracker(
        key=key,
        max_window_size_seconds=max_window_size_seconds,
    )

    # ensure callback is not synchronous
    if not asyncio.iscoroutinefunction(callback):
        raise ValueError("callback must be an asynchronous function")

    def initialize_and_get_window(session_):
        cycle_tracker.initialize(session_)
        return cycle_tracker.get_window(session_)

    try:
        async with framework.db.session.get_db_session_async() as session:
            last_update_time = await mlrun.utils.run_in_threadpool(
                initialize_and_get_window, session
            )
            now = datetime.datetime.now(datetime.UTC)
            await callback(session, last_update_time, *args, **kwargs)
            await mlrun.utils.run_in_threadpool(
                cycle_tracker.update_window, session, now
            )
        # The window update succeeded above, no need to ensure it
        ensure_window_update = False
    finally:
        if ensure_window_update:
            # Sessions are not thread-safe, so we need to create a new one
            await mlrun.utils.run_in_threadpool(
                framework.db.session.run_function_with_new_db_session,
                cycle_tracker.update_window,
                now,
            )


def run_with_time_window_tracker_sync(
    key: TimeWindowTrackerKeys,
    max_window_size_seconds: int,
    callback: typing.Callable,
    *args,
    **kwargs,
):
    """
    Synchronous version of run_with_time_window_tracker with some differences:
    1. This function reduces the overhead of running synchronous code in an async context by avoiding unnecessary
    thread switching.
    2. No need for ensure_window_update parameter.

    NOTE: Use this function when your callback is synchronous.
    """
    cycle_tracker = TimeWindowTracker(
        key=key,
        max_window_size_seconds=max_window_size_seconds,
    )

    # ensure callback is synchronous
    if asyncio.iscoroutinefunction(callback):
        raise ValueError("callback must be a synchronous function")

    with framework.db.session.get_db_session() as session:
        cycle_tracker.initialize(session)
        last_update_time = cycle_tracker.get_window(session)
        now = datetime.datetime.now(datetime.UTC)
        callback(session, last_update_time, *args, **kwargs)
        cycle_tracker.update_window(session, now)
