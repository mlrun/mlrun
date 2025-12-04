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
Tests for TimescaleDB stream graph steps.
"""

import json

from mlrun.model_monitoring.db.tsdb.timescaledb.timescaledb_stream_graph_steps import (
    ProcessBeforeTimescaleDB,
)


class TestProcessBeforeTimescaleDB:
    """Tests for ProcessBeforeTimescaleDB graph step."""

    def test_event_fields_populated(self):
        """Test that required fields are populated from event."""
        processor = ProcessBeforeTimescaleDB()

        endpoint_id = "ep1"
        timestamp = "2024-06-15T10:00:00+00:00"
        project = "my_project"
        function = "my_function"
        metrics = {"accuracy": 0.95}

        event = {
            "endpoint_id": endpoint_id,
            "when": timestamp,
            "function_uri": f"{project}/{function}",
            "metrics": metrics,
        }

        result = processor.do(event)

        assert result["project"] == project
        assert result["end_infer_time"] == timestamp
        assert result["custom_metrics"] == json.dumps(metrics)
        assert result["table_column"] == f"_{endpoint_id}"

    def test_timestamp_field_fallback(self):
        """Test that timestamp field is used when 'when' is not present."""
        processor = ProcessBeforeTimescaleDB()

        timestamp = "2024-06-15T11:00:00+00:00"
        event = {
            "endpoint_id": "ep1",
            "timestamp": timestamp,
            "function_uri": "project/function",
            "metrics": {},
        }

        result = processor.do(event)

        assert result["end_infer_time"] == timestamp

    def test_empty_metrics_serialized(self):
        """Test that empty metrics dict is serialized correctly."""
        processor = ProcessBeforeTimescaleDB()

        metrics = {}
        event = {
            "endpoint_id": "ep1",
            "when": "2024-06-15T10:00:00+00:00",
            "function_uri": "project/function",
            "metrics": metrics,
        }

        result = processor.do(event)

        assert result["custom_metrics"] == json.dumps(metrics)

    def test_missing_metrics_handled(self):
        """Test that missing metrics field is handled gracefully."""
        processor = ProcessBeforeTimescaleDB()

        event = {
            "endpoint_id": "ep1",
            "when": "2024-06-15T10:00:00+00:00",
            "function_uri": "project/function",
        }

        result = processor.do(event)

        # Missing metrics should default to empty dict
        assert result["custom_metrics"] == json.dumps({})
