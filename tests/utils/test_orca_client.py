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
import types
import unittest.mock
import uuid

import pytest

import mlrun.errors
import mlrun.utils
import mlrun.utils.orca_client as orca_client


@pytest.mark.parametrize(
    "action_state, expected_failed",
    [
        ("failed", True),
        # never terminal - the poll times out
        ("running", False),
    ],
)
def test_is_action_failed_tells_failure_from_timeout(
    action_state: str, expected_failed: bool
):
    send_request = unittest.mock.Mock(
        return_value=unittest.mock.Mock(
            json=lambda: {"items": [{"status": {"state": action_state}}]}
        )
    )
    orchestrator = orca_client.ProjectsOrchestrator(
        send_request,
        mlrun.utils.logger,
        poll_interval_seconds=0,
        poll_timeout_seconds=0.1,
    )

    with pytest.raises(mlrun.errors.MLRunRetryExhaustedError) as exc_info:
        orchestrator.wait_for_op("p1", str(uuid.uuid4()))

    assert orca_client.is_action_failed(exc_info.value) is expected_failed


@pytest.mark.parametrize(
    "caller_op_id, observed_op_id, expected",
    [
        ("caller-op", "observed-op", "caller-op"),
        (None, "observed-op", "observed-op"),
    ],
)
def test_resolve_prev_op_id_uses_observed_op_id_without_reading(
    caller_op_id: str | None, observed_op_id: str, expected: str
):
    send_request = unittest.mock.Mock()
    orchestrator = orca_client.ProjectsOrchestrator(
        send_request,
        mlrun.utils.logger,
        poll_interval_seconds=0,
        poll_timeout_seconds=0,
    )
    project = types.SimpleNamespace(status=types.SimpleNamespace(op_id=caller_op_id))

    prev_op_id = orchestrator.resolve_prev_op_id(
        "p1", project, observed_op_id=observed_op_id
    )

    assert prev_op_id == expected
    send_request.assert_not_called()
