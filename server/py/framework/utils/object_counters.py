# Copyright 2024 Iguazio
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

import typing

import sqlalchemy.orm

import mlrun.common.schemas

import framework.utils.singletons.db


class ObjectCounter:
    def __init__(
        self,
        project: str,
        object_kind: str,
        object_subkind: typing.Optional[str] = None,
    ):
        self._db = framework.utils.singletons.db.get_db()
        self._project = project
        self._object_kind = object_kind
        self._object_subkind = object_subkind

    def counter(self, db_session: sqlalchemy.orm.Session) -> int:
        """
        Get the current counter value from the database.
        If the counter does not exist, it will be created with a value of 0.
        :param db_session: SQLAlchemy session
        :return: The current counter value
        """
        object_counter = self.get_or_create(db_session)
        return object_counter.counter

    def populate(self, db_session: sqlalchemy.orm.Session, counter: int = 0) -> int:
        """
        Set the counter value in the database.
        :param db_session: SQLAlchemy session
        :param counter: The counter value to set
        :return: The counter value
        """
        object_counter = self._store(db_session, counter)
        return object_counter.counter

    def increment(self, db_session: sqlalchemy.orm.Session) -> int:
        """
        Increment the counter value in the database.
        If the counter does not exist, it will be created with a value of 0 and then incremented to 1.
        :param db_session: SQLAlchemy session
        :return: The new counter value
        """
        object_counter = self.get_or_create(db_session)
        object_counter = self._store(db_session, object_counter.counter + 1)
        return object_counter.counter

    def decrement(self, db_session: sqlalchemy.orm.Session) -> int:
        """
        Decrement the counter value in the database.
        If the counter does not exist, it will be created with a value of 0. In this case or in the case where the
        counter is already 0, the counter will not be decremented.
        :param db_session: SQLAlchemy session
        :return: The new counter value
        """
        object_counter = self.get_or_create(db_session)
        if object_counter.counter > 0:
            object_counter = self._store(db_session, object_counter.counter - 1)
        return object_counter.counter

    def get_or_create(
        self, db_session: sqlalchemy.orm.Session
    ) -> mlrun.common.schemas.ObjectCounter:
        """
        Get or create the counter object from the database.
        If the counter does not exist, it will be created with a value of 0.
        :param db_session: SQLAlchemy session
        :return: The counter object
        """
        return self._db.get_or_create_object_counter(
            db_session, self._project, self._object_kind, self._object_subkind
        )

    def delete(self, db_session: sqlalchemy.orm.Session) -> None:
        """
        Delete the counter from the database.
        :param db_session: SQLAlchemy session
        """
        self._db.delete_object_counter(
            db_session, self._project, self._object_kind, self._object_subkind
        )

    def _store(
        self, db_session: sqlalchemy.orm.Session, counter: int
    ) -> mlrun.common.schemas.ObjectCounter:
        """
        Store the counter object in the database with a given value.
        :param db_session: SQLAlchemy session
        :param counter: The counter value to store
        :return: The counter object
        """
        return self._db.store_object_counter(
            db_session, self._project, self._object_kind, self._object_subkind, counter
        )
