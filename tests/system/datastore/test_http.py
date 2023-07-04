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

import os

import mlrun.datastore
from tests.system.base import TestMLRunSystem


class TestHttpDataStore(TestMLRunSystem):
    def test_https_auth_token_with_env(self):
        mlrun.mlconf.hub_url = (
            "https://raw.githubusercontent.com/mlrun/private-system-tests/"
        )
        os.environ["HTTPS_AUTH_TOKEN"] = os.environ["MLRUN_SYSTEM_TESTS_GIT_TOKEN"]
        func = mlrun.import_function(
            "hub://support_private_hub_repo/func:main",
            secrets=None,
        )
        assert func.metadata.name == "func"

    def test_https_auth_token_with_secrets_flag(self):
        mlrun.mlconf.hub_url = (
            "https://raw.githubusercontent.com/mlrun/private-system-tests/"
        )
        secrets = {"HTTPS_AUTH_TOKEN": os.environ["MLRUN_SYSTEM_TESTS_GIT_TOKEN"]}
        func = mlrun.import_function(
            "hub://support_private_hub_repo/func:main", secrets=secrets
        )
        assert func.metadata.name == "func"
