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
"""
This module provides SQLAlchemy TypeDecorator subclasses that are aware of
database dialects (MySQL, PostgreSQL, SQLite) and automatically select
appropriate native types (e.g., UUID, BLOB, TIMESTAMP with precision) or
fallbacks (e.g., hex-string storage) to ensure consistent behavior across
different database backends.
"""

import uuid
from typing import Any, Optional, Union

import sqlalchemy.types
from sqlalchemy import CHAR, Text
from sqlalchemy.dialects.mysql import DATETIME as MYSQL_DATETIME
from sqlalchemy.dialects.mysql import MEDIUMBLOB
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.dialects.postgresql import TIMESTAMP as PG_TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import TypeDecorator

import mlrun.common.db.dialects


class DateTime(TypeDecorator):
    impl = sqlalchemy.types.DateTime
    cache_ok = True
    precision: int = 3

    def load_dialect_impl(
        self,
        dialect: Dialect,
    ) -> sqlalchemy.types.TypeEngine:
        if dialect.name == mlrun.common.db.dialects.Dialects.MYSQL:
            return dialect.type_descriptor(
                MYSQL_DATETIME(
                    fsp=self.precision,
                    timezone=True,
                )
            )
        if dialect.name == mlrun.common.db.dialects.Dialects.POSTGRESQL:
            return dialect.type_descriptor(
                PG_TIMESTAMP(
                    precision=self.precision,
                    timezone=True,
                )
            )
        return dialect.type_descriptor(sqlalchemy.types.DateTime)


class MicroSecondDateTime(DateTime):
    cache_ok = True
    precision: int = 6


class Blob(TypeDecorator):
    impl = sqlalchemy.types.LargeBinary
    cache_ok = True

    def load_dialect_impl(
        self,
        dialect: Dialect,
    ) -> sqlalchemy.types.TypeEngine:
        if dialect.name == mlrun.common.db.dialects.Dialects.MYSQL:
            return dialect.type_descriptor(MEDIUMBLOB)
        if dialect.name == mlrun.common.db.dialects.Dialects.POSTGRESQL:
            return dialect.type_descriptor(BYTEA)
        return dialect.type_descriptor(self.impl)


class Utf8BinText(TypeDecorator):
    impl = Text
    cache_ok = True

    def load_dialect_impl(
        self,
        dialect: Dialect,
    ) -> sqlalchemy.types.TypeEngine:
        if dialect.name == mlrun.common.db.dialects.Dialects.MYSQL:
            return dialect.type_descriptor(
                sqlalchemy.dialects.mysql.VARCHAR(
                    collation="utf8_bin",
                    length=255,
                )
            )
        if dialect.name == mlrun.common.db.dialects.Dialects.POSTGRESQL:
            # This collation is created as part of the database creation
            return dialect.type_descriptor(
                Text(
                    collation="utf8_bin",
                )
            )
        if dialect.name == mlrun.common.db.dialects.Dialects.SQLITE:
            return dialect.type_descriptor(
                Text(
                    collation="BINARY",
                )
            )
        return dialect.type_descriptor(self.impl)


class UuidType(TypeDecorator):
    """
    A UUID type which stores as native UUID on Postgres (as_uuid=True)
    and as 32-char hex strings on other dialects.
    """

    impl = CHAR(32)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> sqlalchemy.types.TypeEngine:
        if dialect.name == mlrun.common.db.dialects.Dialects.POSTGRESQL:
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(
        self,
        value: Optional[Union[uuid.UUID, str]],
        dialect: Dialect,
    ) -> Optional[Union[uuid.UUID, str]]:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return (
                value
                if dialect.name == mlrun.common.db.dialects.Dialects.POSTGRESQL
                else value.hex
            )
        if isinstance(value, str):
            u = uuid.UUID(value)
            return (
                u
                if dialect.name == mlrun.common.db.dialects.Dialects.POSTGRESQL
                else u.hex
            )
        raise ValueError(f"Cannot bind UUID value {value!r}")

    def process_result_value(
        self, value: Optional[Union[uuid.UUID, bytes, str]], dialect: Dialect
    ) -> Optional[uuid.UUID]:
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(value)

    def coerce_compared_value(self, op: Any, value: Any) -> TypeDecorator:
        # ensure STR comparisons are coerced through this type
        return self
