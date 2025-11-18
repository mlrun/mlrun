# Copyright 2024 Iguazio
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

"""
Integration tests for drain callback behavior during Kafka rebalancing.

Tests the fix for ML-11518: Memory leak in Kafka stream pods caused by
ParquetTarget buffers not being flushed before ACK channels close during rebalancing.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import storey

from mlrun.datastore.storeytargets import ParquetStoreyTarget
from mlrun.serving.server import _flush_all_batching_steps


class TestDrainCallbackFlushBehavior:
    """
    Integration tests for drain callback flush behavior.

    These tests verify that batching steps (ParquetTarget) properly flush
    their buffers when the drain callback is invoked during Kafka rebalancing.
    """

    @pytest.fixture
    def temp_parquet_dir(self):
        """Create temporary directory for Parquet output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_nuclio_context(self):
        """Create minimal mock Nuclio context"""
        context = MagicMock()
        context.logger = MagicMock()
        context.logger.info = MagicMock()
        context.logger.warning = MagicMock()
        context.platform = MagicMock()
        context.platform.set_drain_callback = MagicMock()
        return context

    @pytest.fixture
    async def serving_graph_with_parquet(self, temp_parquet_dir, mock_nuclio_context):
        """Create a real storey flow with actual ParquetTarget for testing"""
        # Create a real async emit source
        source = storey.AsyncEmitSource()

        # Create real ParquetTarget
        parquet_target = ParquetStoreyTarget(
            name="ParquetTarget",
            path=temp_parquet_dir,
            max_events=100,  # Small batch for testing
            flush_after_seconds=300,  # Long timeout - we want to test explicit flush
            infer_columns_from_data=True,
        )

        # Set context on the target
        parquet_target.context = mock_nuclio_context

        # Build the flow: source -> parquet_target
        flow = storey.build_flow([source, parquet_target])

        # Run the flow (returns controller, not a coroutine)
        controller = flow.run()

        # Create a wrapper object that mimics the graph structure
        # that _flush_all_batching_steps() expects
        class FlowWrapper:
            def __init__(self, target, ctrl, src):
                self._outlets = [target]
                self._async_flow = self
                self._controller = src  # Use source for emitting
                self._target = target
                self._flow_controller = ctrl  # Keep flow controller for termination

        wrapper = FlowWrapper(parquet_target, controller, source)

        try:
            yield wrapper
        finally:
            # Ensure flow is terminated even if test fails
            # This prevents orphaned background tasks and file handles
            await controller.terminate()

    @pytest.mark.asyncio
    async def test_flush_all_batching_steps_with_parquet_target(
        self, serving_graph_with_parquet, temp_parquet_dir, mock_nuclio_context
    ):
        """
        Test that _flush_all_batching_steps() correctly flushes ParquetTarget buffers.

        This is the core fix for ML-11518. Uses REAL ParquetTarget and storey flow.
        """
        # Get the real ParquetTarget from the flow
        parquet_target = serving_graph_with_parquet._outlets[0]
        assert (
            parquet_target.name == "ParquetTarget"
        ), "ParquetTarget not found in graph"

        # Send real events through the flow (less than max_events=100 to keep them buffered)
        controller = serving_graph_with_parquet._controller
        test_events = [
            {"value": i, "timestamp": f"2024-01-01T00:00:{i:02d}"} for i in range(50)
        ]

        for event_data in test_events:
            # Wrap in real storey Event and emit through source
            event = storey.Event(body=event_data)
            await controller._emit(event)

        # Give async operations time to process events into the batch
        await asyncio.sleep(0.2)

        # Verify events are actually buffered in the REAL ParquetTarget
        batch_count_before = sum(len(batch) for batch in parquet_target._batch.values())
        assert (
            batch_count_before > 0
        ), f"Expected events to be buffered, but batch is empty. Batch keys: {list(parquet_target._batch.keys())}"

        # Verify no Parquet files written yet (batch not full, timeout not reached)
        parquet_files_before = list(Path(temp_parquet_dir).rglob("*.parquet"))

        # Call the flush function (this is what drain_callback does)
        await _flush_all_batching_steps(
            serving_graph_with_parquet, mock_nuclio_context.logger
        )

        # Give async operations time to complete the flush
        await asyncio.sleep(0.2)

        # Verify buffer is now empty after REAL flush
        batch_count_after = sum(len(batch) for batch in parquet_target._batch.values())
        assert (
            batch_count_after == 0
        ), f"Expected batch to be empty after flush, but has {batch_count_after} events"

        # Verify Parquet files were actually written to disk
        parquet_files_after = list(Path(temp_parquet_dir).rglob("*.parquet"))
        assert len(parquet_files_after) > len(parquet_files_before), (
            f"Expected Parquet files to be written after flush. "
            f"Before: {len(parquet_files_before)}, After: {len(parquet_files_after)}"
        )

        # Verify flush was logged
        mock_nuclio_context.logger.info.assert_called()
        log_calls = [
            str(call) for call in mock_nuclio_context.logger.info.call_args_list
        ]
        assert any(
            "Flushing" in str(call) and "buffered events" in str(call)
            for call in log_calls
        ), f"Expected flush log message. Got: {log_calls}"

    @pytest.mark.asyncio
    async def test_flush_handles_empty_batches_gracefully(
        self, serving_graph_with_parquet, mock_nuclio_context
    ):
        """
        Test that flushing with no buffered events doesn't cause errors.
        Uses REAL ParquetTarget with empty batch.
        """
        # Get the real ParquetTarget - it starts with empty batch
        parquet_target = serving_graph_with_parquet._outlets[0]

        # Verify batch is empty in the REAL ParquetTarget
        batch_count = sum(len(batch) for batch in parquet_target._batch.values())
        assert batch_count == 0, "Batch should start empty"

        # Call flush on empty flow - should not raise any exceptions
        await _flush_all_batching_steps(
            serving_graph_with_parquet, mock_nuclio_context.logger
        )

        # Verify no "flushing X events" was logged (batch was empty)
        log_calls = [
            str(call) for call in mock_nuclio_context.logger.info.call_args_list
        ]
        flush_event_logs = [call for call in log_calls if "buffered events" in call]
        assert len(flush_event_logs) == 0, "Should not log flushing when batch is empty"

    @pytest.mark.asyncio
    async def test_flush_handles_errors_gracefully(
        self, serving_graph_with_parquet, mock_nuclio_context
    ):
        """
        Test that errors during flush are caught and logged as warnings.

        This ensures drain callback doesn't block rebalancing even if flush fails.
        Uses REAL ParquetTarget but mocks _emit_all to simulate failure.
        """
        # Get the real ParquetTarget
        parquet_target = serving_graph_with_parquet._outlets[0]

        # Send a real event to buffer it
        controller = serving_graph_with_parquet._controller
        event = storey.Event(body={"value": 1})
        await controller._emit(event)
        await asyncio.sleep(0.1)

        # Verify event is buffered
        batch_count = sum(len(batch) for batch in parquet_target._batch.values())
        assert batch_count > 0, "Expected event to be buffered"

        # Save original _emit_all and replace with failing version
        original_emit_all = parquet_target._emit_all

        async def failing_emit_all():
            raise RuntimeError("Simulated flush failure")

        parquet_target._emit_all = failing_emit_all

        # Call flush - should catch exception and log warning
        await _flush_all_batching_steps(
            serving_graph_with_parquet, mock_nuclio_context.logger
        )

        # Verify warning was logged
        mock_nuclio_context.logger.warning.assert_called()
        warning_calls = [
            str(call) for call in mock_nuclio_context.logger.warning.call_args_list
        ]
        assert any("Error flushing" in str(call) for call in warning_calls)

        # Restore original method for cleanup
        parquet_target._emit_all = original_emit_all

    def test_integration_with_v2_serving_init(self, mock_nuclio_context, monkeypatch):
        """
        Test that drain callback is properly registered during v2_serving_init.

        This verifies the integration point where our fix is applied.
        """
        # Mock the serving spec
        serving_spec = {
            "kind": "serving",
            "spec": {
                "graph": {
                    "kind": "router",
                    "routes": {},
                }
            },
        }

        # Mock get_serving_spec
        monkeypatch.setattr("mlrun.utils.get_serving_spec", lambda: serving_spec)

        # Mock GraphServer.from_dict
        mock_server = MagicMock()
        mock_server.graph = MagicMock()
        mock_server.graph._async_flow = MagicMock()
        mock_server.wait_for_completion = AsyncMock()

        monkeypatch.setattr(
            "mlrun.serving.server.GraphServer.from_dict", lambda spec: mock_server
        )

        # Call v2_serving_init
        from mlrun.serving.server import v2_serving_init

        v2_serving_init(mock_nuclio_context)

        # Verify drain callback was registered
        mock_nuclio_context.platform.set_drain_callback.assert_called_once()

        # Get the registered callback
        drain_callback = mock_nuclio_context.platform.set_drain_callback.call_args[0][0]

        # Verify it's an async function
        assert asyncio.iscoroutinefunction(drain_callback)

        # Verify callback name (should be 'drain_callback')
        assert drain_callback.__name__ == "drain_callback"
