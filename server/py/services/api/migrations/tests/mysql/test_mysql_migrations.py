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
pytest_plugins = [
    "services.api.migrations.tests.base.conftest",
    "services.api.migrations.tests.base.migrations_tests",
    "services.api.migrations.tests.mysql.conftest",
]


from services.api.migrations.tests.base.migrations_tests import (  # noqa
    test_model_definitions_match_ddl,
    test_single_head_revision,
    test_up_down_consistency,
    test_upgrade,
    test_notification_params_to_secret_params,
)

if __name__ == "__main__":
    pass
