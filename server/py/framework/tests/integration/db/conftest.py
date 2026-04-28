# Copyright 2026 Iguazio
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
import secrets

import pytest
import pytest_mock_resources


@pytest.fixture(scope="session")
def complex_mysql_password() -> str:
    return _generate_complex_mysql_password()


@pytest.fixture(scope="session")
def pmr_mysql_config(complex_mysql_password) -> pytest_mock_resources.MysqlConfig:
    # Override the parent integration conftest so this directory's tests run
    # against a MySQL container whose root password contains every char the
    # backup/restore code paths must escape correctly.
    return pytest_mock_resources.MysqlConfig(
        image=os.getenv("MLRUN_MYSQL_IMAGE", "gcr.io/iguazio/mlrun-mysql:8.4"),
        port=3306,
        username="root",
        password=complex_mysql_password,
        root_database="mlrun",
    )


def _generate_complex_mysql_password() -> str:
    # Always include the chars that have historically broken DBBackupUtil at
    # one of its escaping layers — if the random suffix happens to skip them
    # the test would silently lose coverage.
    #   @ : # ?  URL/DSN reserved characters (must be percent-encoded)
    #   $ & | ;  shell metacharacters (would break a naive shell command)
    # Excluded chars (and why) — the MySQL container entrypoint truncates
    # the stored root password if any of these appear in MYSQL_ROOT_PASSWORD:
    #   '   closes IDENTIFIED BY 'pwd'
    #   "   closes the [client] password="…" file the entrypoint writes
    #   \   triggers MySQL string-escape interpretation
    # Backslash- and double-quote-escaping in our own option file is
    # exercised by the unit tests.
    required = "@:#?$&|;"
    return required + secrets.token_urlsafe(8)
