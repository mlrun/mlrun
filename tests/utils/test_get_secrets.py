# Copyright 2022 Iguazio
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
import os

import mlrun
from mlrun.secrets import SecretsStore


def reverser(key):
    return key[::-1]


def test_get_secret_from_env():
    key = "SOME_KEY"
    value = "SOME_VALUE"
    project_secret_value = "SOME_OTHER_VALUE"
    override_value = "SOME_OVERRIDE"

    # Use an env variable
    os.environ[key] = value
    assert mlrun.get_secret_or_env(key) == value

    os.environ[
        SecretsStore.k8s_env_variable_name_for_secret(key)
    ] = project_secret_value
    # Project secrets should not override directly set env variables
    assert mlrun.get_secret_or_env(key) == value

    del os.environ[key]
    assert mlrun.get_secret_or_env(key) == project_secret_value

    # Use a local override dictionary
    local_secrets = {key: override_value}
    assert mlrun.get_secret_or_env(key, secret_provider=local_secrets) == override_value

    # Use a callable
    assert mlrun.get_secret_or_env(key, secret_provider=reverser) == reverser(key)

    # Use a SecretsStore
    store = SecretsStore()
    store.add_source("inline", local_secrets)
    assert mlrun.get_secret_or_env(key, secret_provider=store) == override_value

    # Verify that default is used if nothing else is found
    assert (
        mlrun.get_secret_or_env(
            "SOME_GIBBERISH",
            secret_provider=store,
            default="not gibberish",
        )
        == "not gibberish"
    )
