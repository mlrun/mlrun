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
from collections.abc import Iterator
from pathlib import Path

import pytest

import mlrun
from mlrun.model_monitoring.db._stats import (
    ModelMonitoringCurrentStatsFile,
    ModelMonitoringDriftMeasuresFile,
)
from tests.assets.log_function import features


@pytest.fixture(autouse=True)
def _patch_store_prefixes(tmpdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "MLRUN_MODEL_ENDPOINT_MONITORING__STORE_PREFIXES__DEFAULT",
        f"file://{tmpdir}/users/pipelines/{{project}}/model-endpoints/{{kind}}",
    )
    mlrun.mlconf.reload()


@pytest.fixture
def current_stats_file() -> Iterator[ModelMonitoringCurrentStatsFile]:
    file = ModelMonitoringCurrentStatsFile(project="stats-test", endpoint_id="1")
    file.create()
    yield file
    file.delete()


@pytest.fixture
def current_stats() -> dict:
    current_stats_dictionary = {
        "data": features,
        "timestamp": mlrun.utils.datetime_min(),
    }
    return current_stats_dictionary


@pytest.fixture
def drift_measures_file() -> Iterator[ModelMonitoringCurrentStatsFile]:
    file = ModelMonitoringDriftMeasuresFile(project="stats-test", endpoint_id="1")
    file.create()
    yield file
    file.delete()


@pytest.fixture
def drift_measures() -> dict:
    drift_measure_dictionary = {
        "data": features,
        "timestamp": mlrun.utils.datetime_min(),
    }
    return drift_measure_dictionary


def test_create_current_stats_file():
    file = ModelMonitoringCurrentStatsFile(
        project="project-a", endpoint_id="fdshgffjt5"
    )
    file.create()
    file_content_data, file_content_ts = file.read()
    file.delete()
    assert (
        file_content_data == {}
    ), "Current stats file should be empty on creation expected {}"


def test_create_drift_measure_file():
    file = ModelMonitoringDriftMeasuresFile(
        project="project-a", endpoint_id="fdshgffjt5"
    )
    file.create()
    file_content_data, file_content_ts = file.read()
    file.delete()
    assert (
        file_content_data == {}
    ), "Current stats file should be empty on creation expected {}"


def test_delete_current_stats_file():
    file = ModelMonitoringCurrentStatsFile(
        project="project-b", endpoint_id="dgfgdgrth6346"
    )
    file.create()
    file.delete()
    assert not file._fs.exists(file._path), "The current stats file was not deleted"


def test_delete_drift_measure_file():
    file = ModelMonitoringDriftMeasuresFile(
        project="project-b", endpoint_id="dgfgdgrth6346"
    )
    file.create()
    file.delete()
    assert not file._fs.exists(file._path), "The current stats file was not deleted"


def test_current_stats(
    current_stats_file: ModelMonitoringCurrentStatsFile, current_stats: dict
):
    current_stats_file.create()
    current_stats_file.write(*current_stats.values())
    stats_data, timestamp = current_stats_file.read()
    assert (
        current_stats["data"] == stats_data and current_stats["timestamp"] == timestamp
    ), "Wrong fetched data from current stats file"


def test_drift_measure(
    drift_measures_file: ModelMonitoringDriftMeasuresFile, drift_measures: dict
):
    drift_measures_file.create()
    drift_measures_file.write(*drift_measures.values())
    stats_data, timestamp = drift_measures_file.read()
    assert (
        drift_measures["data"] == stats_data
        and drift_measures["timestamp"] == timestamp
    ), "Wrong fetched data from current stats file"
