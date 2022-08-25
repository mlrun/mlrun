# Copyright 2018 Iguazio
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
from mlrun.api.utils.db.mysql import MySQLUtil

# fmt: off
mysql_dsn_data = MySQLUtil.get_mysql_dsn_data()
if mysql_dsn_data:
    from .models_mysql import *  # noqa

    # importing private variables as well
    from .models_mysql import _classes, _labeled, _table2cls, _tagged  # noqa # isort:skip
else:
    from .models_sqlite import *  # noqa

    # importing private variables as well
    from .models_sqlite import _classes, _labeled, _table2cls, _tagged  # noqa # isort:skip
# fmt: on
