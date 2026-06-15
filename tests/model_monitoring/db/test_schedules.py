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
from datetime import UTC, datetime
from pathlib import Path

import pytest

import mlrun
import mlrun.utils
from mlrun.model_monitoring.db._schedules import (
    ModelMonitoringSchedulesFileApplication,
    ModelMonitoringSchedulesFileChief,
    ModelMonitoringSchedulesFileEndpoint,
    delete_model_monitoring_schedules_folder,
    delete_model_monitoring_schedules_user_folder,
)
from mlrun.model_monitoring.helpers import (
    _get_monitoring_schedules_folder_path,
    _get_monitoring_schedules_user_folder_path,
)


@pytest.fixture(autouse=True)
def _patch_store_prefixes(tmpdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "MLRUN_MODEL_ENDPOINT_MONITORING__STORE_PREFIXES__DEFAULT",
        f"file://{tmpdir}/users/pipelines/{{project}}/model-endpoints/{{kind}}",
    )
    mlrun.mlconf.reload()


class TestModelMonitoringSchedulesFileEndpoint:
    @pytest.fixture
    def schedules_file(self) -> Iterator[ModelMonitoringSchedulesFileEndpoint]:
        f = ModelMonitoringSchedulesFileEndpoint(
            project="project-0", endpoint_id="endpoint-0"
        )
        f.create()
        yield f
        f.delete()

    def test_create_file(self) -> None:
        file = ModelMonitoringSchedulesFileEndpoint(
            project="abc", endpoint_id="reoko1220a"
        )
        file.create()
        assert file._item.get().decode() == "{}", (
            "The newly created schedules file is different than expected"
        )

    def test_delete_non_existent_file(self) -> None:
        ModelMonitoringSchedulesFileEndpoint(
            project="p0", endpoint_id="ep-1-without-file"
        ).delete()

    def test_delete_file(self) -> None:
        file = ModelMonitoringSchedulesFileEndpoint(
            project="p1", endpoint_id="ep-1-with-file"
        )
        file.create()
        file.delete()
        assert not file._fs.exists(file._path), "The schedules file wasn't deleted"

    def test_unique_last_analyzed_per_app(
        self,
        schedules_file: ModelMonitoringSchedulesFileEndpoint,
    ) -> None:
        app1_name = "app-A"
        app1_last_analyzed = 1716720842
        app2_name = "app-B"

        with schedules_file as f:
            f.update_application_time(
                application=app1_name, timestamp=app1_last_analyzed
            )

            assert f.get_application_time(app1_name) == app1_last_analyzed
            assert f.get_application_time(app2_name) is None

    def test_stored_last_analyzed(
        self,
        schedules_file: ModelMonitoringSchedulesFileEndpoint,
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

    def test_file_not_opened_error(
        self,
        schedules_file: ModelMonitoringSchedulesFileEndpoint,
    ) -> None:
        with pytest.raises(
            mlrun.errors.MLRunValueError,
            match="Open the schedules file as a context manager first",
        ):
            schedules_file.get_application_time(application="my-app")

    def test_not_found_error(self) -> None:
        with pytest.raises(FileNotFoundError):
            with ModelMonitoringSchedulesFileEndpoint(
                project="project-0", endpoint_id="endpoint-0"
            ):
                pass


class TestModelMonitoringSchedulesFileChief:
    @pytest.fixture
    def schedules_file(self) -> Iterator[ModelMonitoringSchedulesFileChief]:
        f = ModelMonitoringSchedulesFileChief(
            project="project-1",
        )
        f.create()
        yield f
        f.delete()

    def test_create_file(self) -> None:
        file = ModelMonitoringSchedulesFileChief(
            project="abc",
        )
        file.create()
        assert file._item.get().decode() == "{}", (
            "The newly created schedules file is different than expected"
        )

    def test_delete_non_existent_file(self) -> None:
        ModelMonitoringSchedulesFileChief(
            project="p1",
        ).delete()

    def test_delete_file(self) -> None:
        file = ModelMonitoringSchedulesFileChief(
            project="p1",
        )
        file.create()
        file.delete()
        assert not file._fs.exists(file._path), "The schedules file wasn't deleted"

    def test_stored_times(
        self,
        schedules_file: ModelMonitoringSchedulesFileChief,
    ) -> None:
        mep1_name = "app-A"
        mep1_last_analyzed = 1716720842.0
        mep1_last_request = 1716720841.0

        with schedules_file as f:
            f.update_endpoint_timestamps(
                endpoint_uid=mep1_name,
                last_request=mep1_last_request,
                last_analyzed=mep1_last_analyzed,
            )

            assert f.get_endpoint_last_request(mep1_name) == mep1_last_request
            assert f.get_endpoint_last_analyzed(mep1_name) is mep1_last_analyzed

    def test_file_not_opened_error(
        self,
        schedules_file: ModelMonitoringSchedulesFileChief,
    ) -> None:
        with pytest.raises(
            mlrun.errors.MLRunValueError,
            match="Open the schedules file as a context manager first",
        ):
            schedules_file.get_endpoint_last_request(endpoint_uid="my-mep")

    def test_not_found_error(self) -> None:
        with pytest.raises(FileNotFoundError):
            with ModelMonitoringSchedulesFileChief(
                project="project-0",
            ):
                pass

    def test_get_or_create(self):
        my_mep_last_request = 1716720841
        my_mep_last_analyzed = 1716720842

        ModelMonitoringSchedulesFileChief(project="project-1").get_or_create()
        with ModelMonitoringSchedulesFileChief(project="project-1") as f:
            f.update_endpoint_timestamps(
                endpoint_uid="my-mep",
                last_request=my_mep_last_request,
                last_analyzed=my_mep_last_analyzed,
            )

        ModelMonitoringSchedulesFileChief(project="project-1").get_or_create()
        with ModelMonitoringSchedulesFileChief(project="project-1") as f1:
            assert (
                f1.get_endpoint_last_request(endpoint_uid="my-mep")
                == my_mep_last_request
                and f1.get_endpoint_last_analyzed(endpoint_uid="my-mep")
                == my_mep_last_analyzed
            )
            f1.delete()


class TestModelMonitoringSchedulesFileApplication:
    @staticmethod
    @pytest.fixture
    def out_path(tmpdir: Path) -> str:
        return str(tmpdir)

    @staticmethod
    def test_model_endpoint(out_path: str) -> None:
        file = ModelMonitoringSchedulesFileApplication(
            out_path=out_path, application="app1"
        )

        ep1_uid = "aknak2s"
        ep2_uid = "9339rkd"
        dt1 = datetime(2020, 10, 2, 1, tzinfo=UTC)
        dt2 = datetime(2020, 10, 2, 2, tzinfo=UTC)

        with file:
            assert file.get_endpoint_last_analyzed(ep1_uid) is None
            file.update_endpoint_last_analyzed(ep1_uid, dt1)
            assert file.get_endpoint_last_analyzed(ep1_uid) == dt1
            file.update_endpoint_last_analyzed(ep2_uid, dt1)
            file.update_endpoint_last_analyzed(ep1_uid, dt2)
            assert file.get_endpoint_last_analyzed(ep1_uid) == dt2
            assert file.get_endpoint_last_analyzed(ep2_uid) == dt1


def test_delete_non_existent_folder() -> None:
    delete_model_monitoring_schedules_folder("proj-without-any-mep")


def test_delete_folder() -> None:
    project = "monitored-endpoints"
    for endpoint_id in ("ep-1", "ep-2", "ep-3"):
        file = ModelMonitoringSchedulesFileEndpoint(
            project=project, endpoint_id=endpoint_id
        )
        file.create()
        filesystem = file._fs

    delete_model_monitoring_schedules_folder(project)
    assert not filesystem.exists(_get_monitoring_schedules_folder_path(project)), (
        "Schedules folder should have been removed"
    )


@pytest.fixture
def _path_artifact_path(tmpdir: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    artifact_path = f"file://{tmpdir}/projects/{{project}}/artifacts"
    monkeypatch.setenv("MLRUN_ARTIFACT_PATH", artifact_path)
    mlrun.mlconf.reload()


@pytest.mark.usefixtures("_path_artifact_path")
def test_delete_user_application_folder() -> None:
    project = "monitored-endpoints"
    out_path = mlrun.utils.helpers.template_artifact_path(
        mlrun.mlconf.artifact_path, project=project
    )
    for application in ("app-1", "app_2"):
        file = ModelMonitoringSchedulesFileApplication(
            out_path=out_path, application=application
        )
        file.create()
        filesystem = file._fs

    delete_model_monitoring_schedules_user_folder(project)
    assert not filesystem.exists(_get_monitoring_schedules_user_folder_path(project)), (
        "Schedules folder should have been removed"
    )
