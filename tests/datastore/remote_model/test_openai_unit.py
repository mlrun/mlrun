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

import concurrent.futures
import math
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
        """Ensure batch_invoke caps concurrent work to openai_batch_max_workers_per_batch."""
        latency = 0.1

        per_batch_limit = mlrun.mlconf.model_providers.openai_batch_max_workers
        global_limit = mlrun.mlconf.model_providers.openai_batch_max_workers_global
        total_messages = global_limit

        effective_parallelism = max(1, min(per_batch_limit, global_limit))

        with unittest.mock.patch(
            "mlrun.datastore.model_provider.openai_provider.OpenAIProvider._single_invoke",
            mock_single_invoke,
        ):
            provider = mlrun.get_model_provider(
                url="openai://gpt-4o-mini",
                secrets={"OPENAI_API_KEY": "test-key"},
            )

            messages_list = [
                [{"role": "user", "content": f"message {i}"}]
                for i in range(total_messages)
            ]

            start = time.perf_counter()
            results = provider.invoke(messages=messages_list)
            duration = time.perf_counter() - start

        state = mock_single_invoke.state
        assert len(results) == total_messages
        assert state["call_count"] == total_messages
        assert state["max_concurrent_observed"] <= effective_parallelism

        # Expected duration scales with achievable parallelism.
        expected_duration = (total_messages / effective_parallelism) * latency
        upper_bound = (
            expected_duration + 0.2
        )  # allow some extra time for scheduling delays
        assert expected_duration <= duration <= upper_bound

    def test_sync_global_concurrency_limit(self, mock_single_invoke):
        per_batch_limit = mlrun.mlconf.model_providers.openai_batch_max_workers
        global_limit = mlrun.mlconf.model_providers.openai_batch_max_workers_global
        batches_count = math.ceil(global_limit / per_batch_limit) + 1
        total_messages = batches_count * per_batch_limit
        with unittest.mock.patch(
            "mlrun.datastore.model_provider.openai_provider.OpenAIProvider._single_invoke",
            mock_single_invoke,
        ):
            provider = mlrun.get_model_provider(
                url="openai://gpt-4o-mini",
                secrets={"OPENAI_API_KEY": "test-key"},
            )

            batches = [
                [
                    [{"role": "user", "content": f"batch{b}-msg{i}"}]
                    for i in range(per_batch_limit)
                ]
                for b in range(batches_count)
            ]

            def _run_batch(msgs):
                return provider.invoke(messages=msgs)

            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=batches_count
            ) as executor:
                futures = [executor.submit(_run_batch, batch) for batch in batches]
                results_per_batch = [f.result() for f in futures]
            duration = time.perf_counter() - start

        state = mock_single_invoke.state

        assert len(results_per_batch) == batches_count
        assert all(
            len(batch_results) == per_batch_limit for batch_results in results_per_batch
        )
        # Total calls across all batches
        assert state["call_count"] == total_messages
        # Global limit ensures max concurrent across all batches
        assert state["max_concurrent_observed"] <= global_limit

        latency = 0.1
        expected_duration = math.ceil(total_messages / global_limit) * latency
        upper_bound = expected_duration + 0.1
        assert expected_duration <= duration <= upper_bound
