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

from os import environ

from mlrun.secrets import SecretsStore

spec = {
    "secret_sources": [
        {"kind": "file", "source": "tests/secrets_test.txt"},
        {"kind": "inline", "source": {"abc": "def"}},
        {"kind": "env", "source": "ENV123,ENV456"},
    ],
}


def test_load():
    environ["ENV123"] = "xx"
    environ["ENV456"] = "yy"
    ss = SecretsStore.from_list(spec["secret_sources"])

    assert ss.get("ENV123") == "xx", "failed on 1st env var secret"
    assert ss.get("ENV456") == "yy", "failed on 1st env var secret"
    assert ss.get("MYENV") == "123", "failed on 1st env var secret"
    assert ss.get("MY2NDENV") == "456", "failed on 1st env var secret"
    assert ss.get("abc") == "def", "failed on 1st env var secret"
    print(ss.items())


def test_inline_str():
    spec = {
        "secret_sources": [{"kind": "inline", "source": "{'abc': 'def'}"}],
    }

    ss = SecretsStore.from_list(spec["secret_sources"])
    assert ss.get("abc") == "def", "failed on 1st env var secret"
