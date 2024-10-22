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

from pathlib import Path

import pytest

import mlrun
from mlrun.model_monitoring.db._schedules import (
    ModelMonitoringSchedulesFile,
    delete_model_monitoring_schedules_folder,
)
from mlrun.model_monitoring.helpers import _get_monitoring_schedules_folder_path


@pytest.fixture(autouse=True)
def _patch_store_prefixes(tmpdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "MLRUN_MODEL_ENDPOINT_MONITORING__STORE_PREFIXES__DEFAULT",
        f"file://{tmpdir}/users/pipelines/{{project}}/model-endpoints/{{kind}}",
    )
    mlrun.mlconf.reload()


def test_create_file() -> None:
    file = ModelMonitoringSchedulesFile(project="abc", endpoint_id="reoko1220a")
    file.create()
    assert (
        file._item.get().decode() == "{}"
    ), "The newly created schedules file is different than expected"


def test_delete_non_existent_file() -> None:
    ModelMonitoringSchedulesFile(project="p0", endpoint_id="ep-1-without-file").delete()


def test_delete_file() -> None:
    file = ModelMonitoringSchedulesFile(project="p1", endpoint_id="ep-1-with-file")
    file.create()
    file.delete()
    assert not file._fs.exists(file._path), "The schedules file wasn't deleted"


def test_delete_non_existent_folder() -> None:
    delete_model_monitoring_schedules_folder("proj-without-any-mep")


def test_delete_folder() -> None:
    project = "monitored-endpoints"
    for endpoint_id in ("ep-1", "ep-2", "ep-3"):
        file = ModelMonitoringSchedulesFile(project=project, endpoint_id=endpoint_id)
        file.create()
        filesystem = file._fs

    delete_model_monitoring_schedules_folder(project)
    assert not filesystem.exists(
        _get_monitoring_schedules_folder_path(project)
    ), "Schedules folder should have been removed"
