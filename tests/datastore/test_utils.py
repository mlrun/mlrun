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

import json

import kafka
import pytest

import mlrun.datastore
import mlrun.datastore.wasbfs
from mlrun.datastore.utils import KafkaParameters, transform_list_filters_to_tuple


@pytest.mark.parametrize(
    "additional_filters, message",
    [
        ([("x", "=", 3)], ""),
        (
            [[("x", "=", 3), ("x", "=", 4), ("x", "=", 5)]],
            "additional_filters does not support nested list inside filter tuples except in -in- logic.",
        ),
        (
            [[("x", "=", 3), ("x", "=", 4)]],
            "additional_filters does not support nested list inside filter tuples except in -in- logic.",
        ),
        (("x", "=", 3), "mlrun supports additional_filters only as a list of tuples."),
        ([("x", "in", [3, 4]), ("y", "in", [3, 4])], ""),
        ([0], "mlrun supports additional_filters only as a list of tuples."),
        (
            [("age", "=", float("nan"))],
            "using NaN in additional_filters is not supported",
        ),
        (
            [("age", "in", [10, float("nan")])],
            "using NaN in additional_filters is not supported",
        ),
        ([("x", "=", "=", 3), ("y", "in", [3, 4])], "illegal filter tuple length"),
        ([()], ""),
        ([], ""),
    ],
)
def test_transform_list_filters_to_tuple(additional_filters, message):
    back_from_json_serialization = json.loads(json.dumps(additional_filters))

    if message:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match=message):
            transform_list_filters_to_tuple(additional_filters)
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match=message):
            transform_list_filters_to_tuple(
                additional_filters=back_from_json_serialization
            )
    else:
        transform_list_filters_to_tuple(additional_filters)
        result = transform_list_filters_to_tuple(back_from_json_serialization)
        assert result == additional_filters


class TestKafkaParameters:
    def test_producer_option_round_trips(self):
        assert "acks" in kafka.KafkaProducer.DEFAULT_CONFIG
        assert KafkaParameters({"acks": 1, "linger_ms": 5}).producer() == {
            "acks": 1,
            "linger_ms": 5,
        }

    def test_consumer_option_round_trips(self):
        kwargs = {
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,
            "max_poll_records": 100,
        }
        for key in kwargs:
            assert key in kafka.KafkaConsumer.DEFAULT_CONFIG
        assert KafkaParameters(kwargs).consumer() == kwargs

    def test_admin_option_round_trips(self):
        assert "request_timeout_ms" in kafka.KafkaAdminClient.DEFAULT_CONFIG
        assert KafkaParameters({"request_timeout_ms": 30000}).admin() == {
            "request_timeout_ms": 30000
        }

    def test_sasl_expands_into_consumer_config(self):
        config = KafkaParameters(
            {"sasl": {"mechanism": "PLAIN", "user": "u", "password": "p"}}
        ).consumer()
        assert config == {
            "security_protocol": "SASL_PLAINTEXT",
            "sasl_mechanism": "PLAIN",
            "sasl_plain_username": "u",
            "sasl_plain_password": "p",
        }

    def test_validate_keys_raises_on_unknown_key(self):
        with pytest.raises(ValueError, match="not_a_real_kafka_key"):
            KafkaParameters({"not_a_real_kafka_key": 1})

    def test_valid_entries_only_drops_unknown_keys(self):
        assert KafkaParameters({}).valid_entries_only(
            {"acks": 1, "not_a_real_kafka_key": 1}
        ) == {"acks": 1}
