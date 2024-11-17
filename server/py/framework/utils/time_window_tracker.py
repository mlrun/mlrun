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
#
import datetime
import typing

import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

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
        max_window_size_seconds: typing.Optional[int] = None,
    ):
        self._key = key
        self._timestamp = None
        self._max_window_size_seconds = max_window_size_seconds

        self._db = framework.utils.singletons.db.get_db()

    def initialize(self, session: sqlalchemy.orm.Session):
        time_window_tracker_record = self._refresh_from_db(
            session, raise_on_not_found=False
        )
        self._timestamp = self._timestamp or datetime.datetime.now(
            datetime.timezone.utc
        )
        if not time_window_tracker_record:
            self._db.store_time_window_tracker_record(
                session, self._key, self._timestamp, self._max_window_size_seconds
            )

    def update_window(
        self,
        session: sqlalchemy.orm.Session,
        timestamp: typing.Optional[datetime.datetime] = None,
    ):
        self._timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc)
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
            tzinfo=datetime.timezone.utc
        )
        self._max_window_size_seconds = (
            time_window_tracker_record.max_window_size_seconds
        )
        if time_window_tracker_record.max_window_size_seconds is not None:
            self._timestamp = max(
                self._timestamp,
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(seconds=self._max_window_size_seconds),
            )
            self.update_window(session, self._timestamp)

        return time_window_tracker_record


async def run_with_time_window_tracker(
    db_session: sqlalchemy.orm.Session,
    key: TimeWindowTrackerKeys,
    max_window_size_seconds: int,
    ensure_window_update: bool,
    callback: typing.Callable,
    *args,
    **kwargs,
):
    cycle_tracker = TimeWindowTracker(
        key=key,
        max_window_size_seconds=max_window_size_seconds,
    )
    # Although the methods below would not be using the db_session in parallel, for some reason, reusing it
    # causes a segmentation fault so we create new ones for the time window ops
    await run_in_threadpool(
        framework.db.session.run_function_with_new_db_session, cycle_tracker.initialize
    )
    last_update_time = await run_in_threadpool(
        framework.db.session.run_function_with_new_db_session, cycle_tracker.get_window
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    if ensure_window_update:
        try:
            await framework.utils.asyncio.maybe_coroutine(
                callback(db_session, last_update_time, *args, **kwargs)
            )
        finally:
            await run_in_threadpool(
                framework.db.session.run_function_with_new_db_session,
                cycle_tracker.update_window,
                now,
            )
    else:
        await framework.utils.asyncio.maybe_coroutine(
            callback(db_session, last_update_time, *args, **kwargs)
        )
        await run_in_threadpool(
            framework.db.session.run_function_with_new_db_session,
            cycle_tracker.update_window,
            now,
        )
