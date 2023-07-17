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

import mlrun.api.rundb.sqldb
import mlrun.db.factory


def test_container_override():
    factory = mlrun.db.factory.RunDBFactory()
    run_db = factory.create_run_db(url="mock://")
    assert isinstance(run_db, mlrun.api.rundb.sqldb.SQLRunDB)
