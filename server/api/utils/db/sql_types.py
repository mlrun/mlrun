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

import sqlalchemy
import sqlalchemy.dialects.mysql

from .mysql import MySQLUtil


class SQLTypesUtil:
    class _Collations:
        # with sqlite we use the default collation
        sqlite = None
        mysql = "utf8mb3_bin"

    class _Timestamp:
        sqlite = sqlalchemy.TIMESTAMP
        mysql = sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3)

    class _Datetime:
        sqlite = sqlalchemy.DATETIME(timezone=True)
        mysql = sqlalchemy.dialects.mysql.DATETIME(timezone=True, fsp=3)

    class _Blob:
        sqlite = sqlalchemy.BLOB
        mysql = sqlalchemy.dialects.mysql.MEDIUMBLOB

    @classmethod
    def collation(cls):
        return cls._return_type(cls._Collations)

    @classmethod
    def timestamp(cls):
        """
        Use `SQLTypesUtil.datetime()` in new columns.
        See ML-6921.
        """
        return cls._return_type(cls._Timestamp)

    @classmethod
    def datetime(cls):
        return cls._return_type(cls._Datetime)

    @classmethod
    def blob(cls):
        return cls._return_type(cls._Blob)

    @staticmethod
    def _return_type(type_cls: type):
        mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
        if mysql_dsn_data:
            return type_cls.mysql

        return type_cls.sqlite
