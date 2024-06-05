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
import sqlalchemy
import sqlalchemy.dialects.mysql

from .mysql import MySQLUtil


class SQLTypesUtil:
    class Collations:
        # with sqlite we use the default collation
        sqlite = None
        mysql = "utf8mb3_bin"

    class Timestamp:
        sqlite = sqlalchemy.TIMESTAMP
        mysql = sqlalchemy.dialects.mysql.TIMESTAMP(fsp=3)

    class Blob:
        sqlite = sqlalchemy.BLOB
        mysql = sqlalchemy.dialects.mysql.MEDIUMBLOB

    @staticmethod
    def collation():
        return SQLTypesUtil._return_type(SQLTypesUtil.Collations)

    @staticmethod
    def timestamp():
        return SQLTypesUtil._return_type(SQLTypesUtil.Timestamp)

    @staticmethod
    def blob():
        return SQLTypesUtil._return_type(SQLTypesUtil.Blob)

    @staticmethod
    def _return_type(type_cls):
        mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
        if mysql_dsn_data:
            return type_cls.mysql

        return type_cls.sqlite
