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

import sqlalchemy.dialects.mysql
import sqlalchemy.dialects.postgresql
import sqlalchemy.types
from sqlalchemy import Text


class DateTime(sqlalchemy.types.TypeDecorator):
    impl = sqlalchemy.types.DateTime
    cache_ok = True
    precision = 3

    def load_dialect_impl(
        self, dialect: sqlalchemy.engine.interfaces.Dialect
    ) -> sqlalchemy.engine.interfaces.Dialect.type_descriptor:
        if dialect.name == "mysql":
            return dialect.type_descriptor(
                sqlalchemy.dialects.mysql.DATETIME(fsp=self.precision, timezone=True)
            )
        elif dialect.name == "postgresql":
            return dialect.type_descriptor(
                sqlalchemy.dialects.postgresql.TIMESTAMP(
                    precision=self.precision, timezone=True
                )
            )
        else:
            return dialect.type_descriptor(sqlalchemy.types.DateTime)


class MicroSecondDateTime(DateTime):
    cache_ok = True
    precision = 6


class Blob(sqlalchemy.types.TypeDecorator):
    impl = sqlalchemy.types.LargeBinary
    cache_ok = True

    def load_dialect_impl(
        self, dialect: sqlalchemy.engine.interfaces.Dialect
    ) -> sqlalchemy.engine.interfaces.Dialect.type_descriptor:
        if dialect.name == "mysql":
            return dialect.type_descriptor(sqlalchemy.dialects.mysql.MEDIUMBLOB)
        elif dialect.name == "postgresql":
            return dialect.type_descriptor(sqlalchemy.dialects.postgresql.BYTEA)
        else:
            return dialect.type_descriptor(self.impl)


class Utf8BinText(sqlalchemy.types.TypeDecorator):
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "mysql":
            # MySQL: full‐Unicode TEXT with utf8_bin collation
            return dialect.type_descriptor(
                sqlalchemy.dialects.mysql.VARCHAR(collation="utf8_bin", length=255)
            )
        elif dialect.name == "postgresql":
            # PostgreSQL: TEXT with a binary‐style collation: utf8_bin is created via sqlalchemy
            return dialect.type_descriptor(Text(collation="utf8_bin"))
        elif dialect.name == "sqlite":
            # SQLite: default TEXT with BINARY collation (case-sensitive)
            return dialect.type_descriptor(Text(collation="BINARY"))
        else:
            # fallback: plain TEXT
            return dialect.type_descriptor(self.impl)
