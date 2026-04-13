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

import pytest

import mlrun.common.schemas

import services.api.utils.events.iguazio as iguazio_events


@pytest.mark.parametrize(
    "action, expected_kind",
    [
        (
            mlrun.common.schemas.SecretEventActions.created,
            iguazio_events.PROJECT_AUTH_SECRET_CREATED,
        ),
        (
            mlrun.common.schemas.SecretEventActions.updated,
            iguazio_events.PROJECT_AUTH_SECRET_UPDATED,
        ),
    ],
)
def test_auth_secret_event_kind_matches_action(action, expected_kind):
    client = iguazio_events.Client.__new__(iguazio_events.Client)
    client.source = "mlrun-api"

    event = client._generate_auth_secret_event(
        username="test-user",
        secret_name="test-secret",
        action=action,
    )

    assert event.kind == expected_kind, (
        f"Expected event kind '{expected_kind}' for action '{action}', "
        f"but got '{event.kind}'"
    )

def test_created_and_updated_have_different_kinds():
    client = iguazio_events.Client.__new__(iguazio_events.Client)
    client.source = "mlrun-api"

    created_event = client._generate_auth_secret_event(
        username="test-user",
        secret_name="test-secret",
        action=mlrun.common.schemas.SecretEventActions.created,
    )
    updated_event = client._generate_auth_secret_event(
        username="test-user",
        secret_name="test-secret",
        action=mlrun.common.schemas.SecretEventActions.updated,
    )

    assert created_event.kind != updated_event.kind, (
        "Created and updated events should have different kinds, "
        f"but both have kind '{created_event.kind}'"
    )
