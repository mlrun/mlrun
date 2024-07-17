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

import server.api.utils.singletons.db


class TimeWindowTracker:
    def __init__(
        self,
        key: str,
        max_window_size_seconds: typing.Optional[int] = None,
    ):
        self._key = key
        self._updated = None
        self._max_window_size_seconds = max_window_size_seconds

        self._db = server.api.utils.singletons.db.get_db()

    def initialize(self, session: sqlalchemy.orm.Session):
        time_window_tracker_record = self._refresh_from_db(
            session, raise_on_not_found=False
        )
        self._updated = self._updated or datetime.datetime.now(datetime.timezone.utc)
        if not time_window_tracker_record:
            self._db.store_time_window_tracker_record(
                session, self._key, self._updated, self._max_window_size_seconds
            )

    def update_window(
        self, session: sqlalchemy.orm.Session, timestamp: datetime.datetime = None
    ):
        self._updated = timestamp or datetime.datetime.now(datetime.timezone.utc)
        self._db.store_time_window_tracker_record(
            session, self._key, self._updated, self._max_window_size_seconds
        )

    def get_window(self, session: sqlalchemy.orm.Session) -> datetime.datetime:
        self._refresh_from_db(session, raise_on_not_found=True)
        return self._updated

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

        self._updated = time_window_tracker_record.updated.replace(
            tzinfo=datetime.timezone.utc
        )
        self._max_window_size_seconds = (
            time_window_tracker_record.max_window_size_seconds
        )
        if time_window_tracker_record.max_window_size_seconds is not None:
            self._updated = max(
                self._updated,
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(seconds=self._max_window_size_seconds),
            )
            self.update_window(session, self._updated)

        return time_window_tracker_record
