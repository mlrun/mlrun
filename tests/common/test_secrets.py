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

import mlrun.common.schemas
import mlrun.common.secrets


def test_list_user_token_secrets_with_non_token_secrets():
    """
    Verify that list_user_token_secrets does not crash when secrets_map
    contains entries stored via store_secret() that lack 'user_id' key.
    """
    provider = mlrun.common.secrets.InMemorySecretProvider()

    # Store a non-token secret (arbitrary dict without user_id/token_name/expiration)
    provider.store_secret(
        "some-config-secret", {"host": "localhost", "port": "5432"}
    )

    # Store a real user token secret
    auth_info = mlrun.common.schemas.AuthInfo()
    auth_info.user_id = "test-user-123"
    provider.store_user_token_secret(
        auth_info=auth_info,
        token_name="my-token",
        token="secret-token-value",
        expiration=3600,
    )

    # This should NOT raise KeyError despite the non-token secret in secrets_map
    result = provider.list_user_token_secrets(user_id="test-user-123")

    assert len(result) == 1
    assert result[0].name == "my-token"
    assert result[0].user_id == "test-user-123"

def test_list_user_token_secrets_only_non_token_secrets():
    """
    Verify that list_user_token_secrets returns empty list when
    only non-token secrets exist in secrets_map.
    """
    provider = mlrun.common.secrets.InMemorySecretProvider()

    # Store only non-token secrets
    provider.store_secret("redis-config", {"url": "redis://localhost:6379"})
    provider.store_secret("db-config", {"connection_string": "postgres://..."})

    # Should return empty list, not crash
    result = provider.list_user_token_secrets(user_id="any-user")
    assert result == []
