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
from .mysql import MySQLUtil


class SQLCollationUtil:
    class Collations:
        # with sqlite we use the default collation
        sqlite = None
        mysql = "utf8_bin"

    @staticmethod
    def collation():
        mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
        if mysql_dsn_data:
            return SQLCollationUtil.Collations.mysql

        return SQLCollationUtil.Collations.sqlite
