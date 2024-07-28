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


class BackgroundTaskKinds:
    db_migrations = "db.migrations"
    project_deletion = "project.deletion.{0}"
    project_deletion_wrapper = "project.deletion.wrapper.{0}"
    function_deletion = "function.deletion.{0}"
