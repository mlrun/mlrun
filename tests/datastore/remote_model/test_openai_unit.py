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

import threading
import time
import unittest.mock

import pytest

import mlrun


class TestOpenAIBatchConcurrency:
    """Test batch invocation concurrency limits using a lightweight mock."""

    @pytest.fixture
    def mock_single_invoke(self):
        state = {
            "current_running": 0,
            "max_concurrent_observed": 0,
            "lock": threading.Lock(),
            "call_count": 0,
        }

        def _mock(self, messages, invoke_response_format, **kwargs):
            with state["lock"]:
                state["current_running"] += 1
                state["call_count"] += 1
                state["max_concurrent_observed"] = max(
                    state["max_concurrent_observed"], state["current_running"]
                )

            # Simulate API latency for a single OpenAI call
            time.sleep(0.1)

            with state["lock"]:
                state["current_running"] -= 1

            return {"mock": "response", "answer": "mocked"}

        _mock.state = state
        return _mock

    def test_sync_batch_concurrency_limit(self, mock_single_invoke):
        # Config: global limit is high enough not to interfere; per-batch is 5
        mlrun.mlconf.model_providers.openai_batch_max_workers_global = 20
        mlrun.mlconf.model_providers.openai_batch_max_workers_per_batch = 5

        with unittest.mock.patch(
            "mlrun.datastore.model_provider.openai_provider.OpenAIProvider._single_invoke",
            mock_single_invoke,
        ):
            provider = mlrun.get_model_provider(
                url="openai://gpt-4o-mini",
                secrets={"OPENAI_API_KEY": "test-key"},
            )

            messages_list = [
                [{"role": "user", "content": f"message {i}"}] for i in range(20)
            ]

            start = time.perf_counter()
            results = provider.invoke(messages=messages_list)
            duration = time.perf_counter() - start

        state = mock_single_invoke.state
        # Basic sanity
        assert len(results) == 20
        assert state["call_count"] == 20

        # Per-batch concurrency should be capped at 5 by the executor
        assert state["max_concurrent_observed"] <= 5

        # Timing: ideal ~0.4s (20 / 5 * 0.1). Allow some tolerance.
        assert 0.3 <= duration <= 0.6
