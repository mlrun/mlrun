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
import re

import pytest

import mlrun.config
from mlrun import new_function
from mlrun.datastore import KafkaSource


def test_kafka_source_with_old_nuclio():
    kafka_source = KafkaSource()
    function = new_function(name="test-function", kind="remote")
    function.spec.min_replicas = 2
    function.spec.max_replicas = 2

    original_nuclio_version = mlrun.config.config.nuclio_version
    mlrun.config.config.nuclio_version = "1.12.9"
    try:
        with pytest.warns(
            UserWarning,
            match="^Detected nuclio version .* To resolve this, please upgrade Nuclio.$",
        ):
            kafka_source.add_nuclio_trigger(function)
    finally:
        mlrun.config.config.nuclio_version = original_nuclio_version

    assert function.spec.min_replicas == 1
    assert function.spec.max_replicas == 1


def test_kafka_source_with_new_nuclio():
    kafka_source = KafkaSource()
    function = new_function(name="test-function", kind="remote")
    function.spec.min_replicas = 2
    function.spec.max_replicas = 2

    original_nuclio_version = mlrun.config.config.nuclio_version
    mlrun.config.config.nuclio_version = "1.12.10"
    try:
        with pytest.warns() as warnings:
            kafka_source.add_nuclio_trigger(function)
            for warning in warnings:
                assert not re.fullmatch(
                    "Detected nuclio version .* To resolve this, please upgrade Nuclio.",
                    str(warning.message),
                )
    finally:
        mlrun.config.config.nuclio_version = original_nuclio_version

    assert function.spec.min_replicas == 2
    assert function.spec.max_replicas == 2
