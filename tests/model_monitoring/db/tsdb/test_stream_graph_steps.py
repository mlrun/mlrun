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

"""
Tests for shared TSDB stream graph steps.
"""

from mlrun.model_monitoring.db.tsdb.stream_graph_steps import DeduplicateSubEvents


class TestDeduplicateSubEvents:
    """
    Tests for DeduplicateSubEvents graph step (ML-11639).

    This step deduplicates sub-events from batch inference before writing to TSDB,
    improving performance for all backends (TimescaleDB, TDEngine, V3IO).
    """

    def test_first_event_passes_through(self):
        """Test that the first event for an endpoint passes through."""
        deduplicator = DeduplicateSubEvents()

        endpoint_id = "ep1"
        event = {
            "endpoint_id": endpoint_id,
            "when": "2024-06-15T10:00:00+00:00",
        }

        result = deduplicator.do(event)

        assert result is not None
        assert result["endpoint_id"] == endpoint_id

    def test_duplicate_sub_events_filtered(self):
        """
        Test that duplicate sub-events from batch inference are filtered (ML-11639).
        Simulates N sub-events with same (endpoint_id, timestamp).
        """
        deduplicator = DeduplicateSubEvents()

        # Simulate 32 sub-events from batch inference - all have same timestamp
        batch_size = 32
        timestamp = "2024-06-15T10:00:00+00:00"
        endpoint_id = "ep_batch"

        results = [
            deduplicator.do(
                {
                    "endpoint_id": endpoint_id,
                    "when": timestamp,
                    "features": [1.0, 2.0, 3.0],
                    "prediction": [0.5],
                }
            )
            for _ in range(batch_size)
        ]
        passed_results = [r for r in results if r is not None]

        # Only 1 should pass through (deduplication)
        assert len(passed_results) == 1
        assert passed_results[0]["endpoint_id"] == endpoint_id

    def test_different_timestamps_all_pass(self):
        """Test that events with different timestamps all pass through."""
        deduplicator = DeduplicateSubEvents()

        endpoint_id = "ep1"
        timestamps = [
            "2024-06-15T10:00:00+00:00",
            "2024-06-15T10:01:00+00:00",
            "2024-06-15T10:02:00+00:00",
        ]

        results = [
            deduplicator.do(
                {
                    "endpoint_id": endpoint_id,
                    "when": ts,
                }
            )
            for ts in timestamps
        ]
        passed_results = [r for r in results if r is not None]

        # All events with different timestamps should pass
        assert len(passed_results) == len(timestamps)

    def test_different_endpoints_same_timestamp_all_pass(self):
        """Test that different endpoints with same timestamp all pass."""
        deduplicator = DeduplicateSubEvents()

        timestamp = "2024-06-15T10:00:00+00:00"
        endpoints = ["ep1", "ep2", "ep3"]

        results = [
            deduplicator.do(
                {
                    "endpoint_id": ep,
                    "when": timestamp,
                }
            )
            for ep in endpoints
        ]
        passed_results = [r for r in results if r is not None]

        # All different endpoints should pass even with same timestamp
        assert len(passed_results) == len(endpoints)
        passed_endpoint_ids = [r["endpoint_id"] for r in passed_results]
        assert passed_endpoint_ids == endpoints

    def test_interleaved_endpoints_dedup_correctly(self):
        """
        Test deduplication with interleaved events from multiple endpoints.
        Each endpoint gets batch inference (N sub-events with same timestamp).
        """
        deduplicator = DeduplicateSubEvents()

        ts1 = "2024-06-15T10:00:00+00:00"
        ts2 = "2024-06-15T10:01:00+00:00"

        # Simulate interleaved events: ep1 batch, ep2 batch, ep1 new batch
        events = [
            # ep1 batch 1 (3 sub-events with same timestamp)
            {"endpoint_id": "ep1", "when": ts1},
            {"endpoint_id": "ep1", "when": ts1},
            {"endpoint_id": "ep1", "when": ts1},
            # ep2 batch (2 sub-events with same timestamp)
            {"endpoint_id": "ep2", "when": ts1},
            {"endpoint_id": "ep2", "when": ts1},
            # ep1 batch 2 - new timestamp (2 sub-events)
            {"endpoint_id": "ep1", "when": ts2},
            {"endpoint_id": "ep1", "when": ts2},
        ]

        results = [deduplicator.do(e) for e in events]
        passed_results = [r for r in results if r is not None]

        # Should pass: ep1@ts1, ep2@ts1, ep1@ts2 = 3 unique (endpoint, timestamp) pairs
        expected_passed = [
            ("ep1", ts1),
            ("ep2", ts1),
            ("ep1", ts2),
        ]
        actual_passed = [(r["endpoint_id"], r["when"]) for r in passed_results]

        assert len(actual_passed) == len(expected_passed)
        assert actual_passed == expected_passed

    def test_timestamp_field_fallback(self):
        """Test that 'timestamp' field is used when 'when' is not present."""
        deduplicator = DeduplicateSubEvents()

        timestamp = "2024-06-15T10:00:00+00:00"
        endpoint_id = "ep1"

        # First event with 'timestamp' field
        result1 = deduplicator.do(
            {
                "endpoint_id": endpoint_id,
                "timestamp": timestamp,
            }
        )

        # Second event with same timestamp - should be filtered
        result2 = deduplicator.do(
            {
                "endpoint_id": endpoint_id,
                "timestamp": timestamp,
            }
        )

        assert result1 is not None
        assert result2 is None

    def test_event_without_timestamp_passes(self):
        """Test that events without timestamp field pass through without dedup."""
        deduplicator = DeduplicateSubEvents()

        # Events without timestamp should all pass (can't deduplicate without timestamp)
        results = [deduplicator.do({"endpoint_id": "ep1", "data": i}) for i in range(5)]
        passed_results = [r for r in results if r is not None]

        assert len(passed_results) == 5

    def test_event_without_endpoint_passes(self):
        """Test that events without endpoint_id field pass through without dedup."""
        deduplicator = DeduplicateSubEvents()

        timestamp = "2024-06-15T10:00:00+00:00"

        # Events without endpoint_id should all pass (can't deduplicate without endpoint)
        results = [deduplicator.do({"when": timestamp, "data": i}) for i in range(5)]
        passed_results = [r for r in results if r is not None]

        assert len(passed_results) == 5
