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
import mlrun.utils
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


@pytest.fixture
def schedules_file() -> Iterator[ModelMonitoringSchedulesFile]:
    f = ModelMonitoringSchedulesFile(project="project-0", endpoint_id="endpoint-0")
    f.create()
    yield f
    f.delete()


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


def test_unique_last_analyzed_per_app(
    schedules_file: ModelMonitoringSchedulesFile,
) -> None:
    app1_name = "app-A"
    app1_last_analyzed = 1716720842
    app2_name = "app-B"

    with schedules_file as f:
        f.update_application_time(application=app1_name, timestamp=app1_last_analyzed)

        assert f.get_application_time(app1_name) == app1_last_analyzed
        assert f.get_application_time(app2_name) is None


def test_stored_last_analyzed(
    schedules_file: ModelMonitoringSchedulesFile,
) -> None:
    application_name = "dummy-app"
    # Try to get last analyzed value, we expect it to be empty
    with schedules_file as f:
        assert f.get_application_time(application=application_name) is None

    # Update the application timestamp record and validate it is stored as expected
    current_time = int(mlrun.utils.datetime_now().timestamp())
    with schedules_file as f:
        f.update_application_time(
            application=application_name,
            timestamp=current_time,
        )

    with schedules_file as f:
        last_analyzed = f.get_application_time(application=application_name)

    assert last_analyzed == current_time


def test_file_not_opened_error(schedules_file: ModelMonitoringSchedulesFile) -> None:
    with pytest.raises(
        mlrun.errors.MLRunValueError,
        match="Open the schedules file as a context manager first",
    ):
        schedules_file.get_application_time(application="my-app")


def test_not_found_error() -> None:
    with pytest.raises(FileNotFoundError):
        with ModelMonitoringSchedulesFile(
            project="project-0", endpoint_id="endpoint-0"
        ):
            pass
