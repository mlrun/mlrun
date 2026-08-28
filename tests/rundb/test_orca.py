# Copyright 2026 Iguazio
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
import unittest.mock
import uuid

import pytest

import mlrun.common.schemas
import mlrun.db.httpdb
import mlrun.db.orca
import mlrun.errors
import mlrun.projects.project

ORCA_API_URL = "https://igz-api.example.com"


def _project_wire_body(name: str, op_id: str, state: str) -> dict:
    return {
        "metadata": {"name": name, "labels": {}, "annotations": {}},
        "spec": {"owner": "jsmith", "description": "desc"},
        "status": {"state": state, "opId": op_id, "updatedAt": None},
    }


def _mlrun_project(name: str, owner: str | None = None) -> mlrun.projects.MlrunProject:
    return mlrun.projects.MlrunProject(
        metadata=mlrun.projects.project.ProjectMetadata(name=name),
        spec=mlrun.projects.project.ProjectSpec(owner=owner),
    )


@pytest.fixture
def orca_client() -> mlrun.db.orca.OrcaProjectsClient:
    mlrun.mlconf.iguazio_api_url = ORCA_API_URL
    db = mlrun.db.httpdb.HTTPRunDB("https://fake-mlrun-url")
    client = mlrun.db.orca.OrcaProjectsClient(db)
    # keep polling fast in tests
    client._poll_interval_seconds = 0.01
    client._poll_timeout_seconds = 1
    return client


class TestOrcaProjectsClient:
    def test_create_project_waits_for_completion(self, requests_mock, orca_client):
        op_id = str(uuid.uuid4())
        requests_mock.post(
            f"{ORCA_API_URL}/api/v1/projects/projects",
            json={"status": {"opId": op_id}},
            status_code=202,
        )
        requests_mock.get(
            f"{ORCA_API_URL}/api/v1/trackable-actions/executions",
            json={"items": [{"status": {"state": "succeeded"}}]},
        )
        requests_mock.get(
            f"{ORCA_API_URL}/api/v1/projects/projects/p1",
            json=_project_wire_body("p1", op_id, "online"),
        )

        project = _mlrun_project("p1", owner="jsmith")
        result = orca_client.create_project(project)

        # regression guard: the public contract must return mlrun.projects.MlrunProject
        # (matching the legacy MLRun-API path), not the internal wire-schema type - op_id is
        # Orca-sync plumbing and deliberately absent from MlrunProject's own status object.
        assert isinstance(result, mlrun.projects.MlrunProject)
        assert result.metadata.name == "p1"
        assert not hasattr(result.status, "op_id")

        create_request = requests_mock.request_history[0]
        assert create_request.json() == {"name": "p1", "owner": "jsmith"}

    def test_create_project_async_returns_op_id_without_polling(
        self, requests_mock, orca_client
    ):
        op_id = str(uuid.uuid4())
        requests_mock.post(
            f"{ORCA_API_URL}/api/v1/projects/projects",
            json={"status": {"opId": op_id}},
            status_code=202,
        )

        result = orca_client.create_project(
            _mlrun_project("p2"), wait_for_completion=False
        )

        assert result == op_id
        assert len(requests_mock.request_history) == 1

    def test_update_project_uses_prev_op_id_from_status(
        self, requests_mock, orca_client
    ):
        op_id = str(uuid.uuid4())
        prev_op_id = uuid.uuid4()
        requests_mock.put(
            f"{ORCA_API_URL}/api/v1/projects/projects/p3",
            json={"status": {"opId": op_id}},
            status_code=202,
        )
        requests_mock.get(
            f"{ORCA_API_URL}/api/v1/trackable-actions/executions",
            json={"items": [{"status": {"state": "succeeded"}}]},
        )
        requests_mock.get(
            f"{ORCA_API_URL}/api/v1/projects/projects/p3",
            json=_project_wire_body("p3", op_id, "online"),
        )

        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="p3"),
            spec=mlrun.common.schemas.ProjectSpec(owner="jsmith"),
            status=mlrun.common.schemas.ProjectStatus(op_id=prev_op_id),
        )
        orca_client.update_project("p3", project)

        put_request = [r for r in requests_mock.request_history if r.method == "PUT"][0]
        assert put_request.json()["prevOpId"] == str(prev_op_id)

    def test_update_project_resolves_prev_op_id_via_get_when_missing(
        self, requests_mock, orca_client
    ):
        op_id = str(uuid.uuid4())
        requests_mock.get(
            f"{ORCA_API_URL}/api/v1/projects/projects/p4",
            json=_project_wire_body("p4", op_id, "online"),
        )
        requests_mock.put(
            f"{ORCA_API_URL}/api/v1/projects/projects/p4",
            json=_project_wire_body("p4", op_id, "online"),
            status_code=200,
        )

        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="p4"),
            spec=mlrun.common.schemas.ProjectSpec(owner="jsmith"),
        )
        orca_client.update_project("p4", project)

        put_request = [r for r in requests_mock.request_history if r.method == "PUT"][0]
        assert put_request.json()["prevOpId"] == op_id

    def test_update_project_synchronous_200_does_not_poll(
        self, requests_mock, orca_client
    ):
        op_id = str(uuid.uuid4())
        requests_mock.put(
            f"{ORCA_API_URL}/api/v1/projects/projects/p5",
            json=_project_wire_body("p5", op_id, "online"),
            status_code=200,
        )

        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="p5"),
            spec=mlrun.common.schemas.ProjectSpec(owner="jsmith"),
            status=mlrun.common.schemas.ProjectStatus(op_id=uuid.uuid4()),
        )
        result = orca_client.update_project("p5", project)

        assert isinstance(result, mlrun.projects.MlrunProject)
        poll_requests = [
            r for r in requests_mock.request_history if "trackable-actions" in r.url
        ]
        assert not poll_requests, "a synchronous 200 has nothing to poll for"

    def test_update_project_async_returns_op_id_even_if_settled_synchronously(
        self, requests_mock, orca_client
    ):
        # regression guard: wait_for_completion=False must always return an op_id, never the
        # project object - even when Orca happens to answer with a synchronous 200 (e.g. all
        # followers acked within the request). The return type must depend only on the caller's
        # wait_for_completion flag, never on a response detail the caller can't predict; a caller
        # relying on getting back an identifier to poll on later must not silently receive a
        # project object instead.
        op_id = str(uuid.uuid4())
        requests_mock.put(
            f"{ORCA_API_URL}/api/v1/projects/projects/p5b",
            json=_project_wire_body("p5b", op_id, "online"),
            status_code=200,
        )

        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="p5b"),
            spec=mlrun.common.schemas.ProjectSpec(owner="jsmith"),
            status=mlrun.common.schemas.ProjectStatus(op_id=uuid.uuid4()),
        )
        result = orca_client.update_project("p5b", project, wait_for_completion=False)

        assert result == op_id

    def test_patch_project_merges_with_current_state(self, requests_mock, orca_client):
        op_id = str(uuid.uuid4())
        current_body = {
            "metadata": {
                "name": "p6",
                "labels": {"team": "ds"},
                "annotations": {"note": "x"},
            },
            "spec": {"owner": "jsmith", "description": "old desc"},
            "status": {"state": "online", "opId": op_id, "updatedAt": None},
        }
        requests_mock.get(
            f"{ORCA_API_URL}/api/v1/projects/projects/p6", json=current_body
        )
        requests_mock.patch(
            f"{ORCA_API_URL}/api/v1/projects/projects/p6",
            json={"status": {"opId": op_id}},
            status_code=202,
        )
        requests_mock.get(
            f"{ORCA_API_URL}/api/v1/trackable-actions/executions",
            json={"items": [{"status": {"state": "succeeded"}}]},
        )

        # only touches description and adds one label - owner, annotations, and the existing
        # "team" label must survive the merge untouched (Orca's PATCH is full-replace, not
        # merge, so a naive pass-through of this partial patch would wipe them - see orca#1059).
        patch_body = {
            "metadata": {"labels": {"env": "prod"}},
            "spec": {"description": "new desc"},
        }
        orca_client.patch_project("p6", patch_body)

        patch_request = [
            r for r in requests_mock.request_history if r.method == "PATCH"
        ][0]
        sent = patch_request.json()
        assert sent["owner"] == "jsmith"
        assert sent["description"] == "new desc"
        assert sent["labels"] == {"team": "ds", "env": "prod"}
        assert sent["annotations"] == {"note": "x"}
        assert sent["prevOpId"] == op_id

    def test_delete_project_waits_for_completion(self, requests_mock, orca_client):
        op_id = str(uuid.uuid4())
        requests_mock.delete(
            f"{ORCA_API_URL}/api/v1/projects/projects/p7",
            json={"status": {"opId": op_id}},
            status_code=202,
        )
        requests_mock.get(
            f"{ORCA_API_URL}/api/v1/trackable-actions/executions",
            json={"items": [{"status": {"state": "succeeded"}}]},
        )

        assert orca_client.delete_project("p7") is None

    def test_delete_project_action_failed_raises_without_retrying(
        self, requests_mock, orca_client
    ):
        op_id = str(uuid.uuid4())
        requests_mock.delete(
            f"{ORCA_API_URL}/api/v1/projects/projects/p8",
            json={"status": {"opId": op_id}},
            status_code=202,
        )
        requests_mock.get(
            f"{ORCA_API_URL}/api/v1/trackable-actions/executions",
            json={"items": [{"status": {"state": "failed"}}]},
        )

        # a failed action is fatal - it should stop polling immediately, not retry until
        # timeout (matches the identical, already-merged-and-CI-passing behavior of
        # server/py/framework/utils/clients/iguazio/v4.py's _wait_for_op for the same case:
        # retry_until_successful wraps a fatal_exceptions hit in MLRunRetryExhaustedError).
        with pytest.raises(mlrun.errors.MLRunRetryExhaustedError):
            orca_client.delete_project("p8")

        poll_requests = [
            r for r in requests_mock.request_history if "trackable-actions" in r.url
        ]
        assert len(poll_requests) == 1

    def test_create_project_poll_timeout_is_not_treated_as_failure(
        self, requests_mock, orca_client
    ):
        op_id = str(uuid.uuid4())
        requests_mock.post(
            f"{ORCA_API_URL}/api/v1/projects/projects",
            json={"status": {"opId": op_id}},
            status_code=202,
        )
        requests_mock.get(
            f"{ORCA_API_URL}/api/v1/trackable-actions/executions",
            json={"items": [{"status": {"state": "running"}}]},
        )

        with pytest.raises(mlrun.errors.MLRunRetryExhaustedError) as exc_info:
            orca_client.create_project(_mlrun_project("p9"))
        assert "still in" in str(exc_info.value)


class TestHTTPRunDBOrcaGate:
    """HTTPRunDB.create_project/store_project/patch_project/delete_project gate on
    _orca_direct_mode() and delegate to OrcaProjectsClient - or fall through to the legacy
    MLRun-API path unchanged - depending on it.
    """

    @pytest.fixture
    def db(self) -> mlrun.db.httpdb.HTTPRunDB:
        return mlrun.db.httpdb.HTTPRunDB("https://fake-mlrun-url")

    def test_ce_mode_falls_through_to_api_call(self, db):
        mlrun.mlconf.httpdb.authentication.mode = "none"
        mlrun.mlconf.iguazio_api_url = ""
        assert db._orca_direct_mode() is False

        with unittest.mock.patch.object(db, "api_call") as mock_api_call:
            mock_api_call.return_value.status_code = 200
            mock_api_call.return_value.json.return_value = _mlrun_project(
                "gated"
            ).to_dict()
            db.create_project(_mlrun_project("gated"))
            assert mock_api_call.called

    def test_orca_direct_mode_without_configured_url_falls_through(self, db):
        mlrun.mlconf.httpdb.authentication.mode = "iguazio-v4"
        mlrun.mlconf.iguazio_api_url = ""
        assert db._orca_direct_mode() is False

    @pytest.mark.parametrize(
        "method_name,args,orca_method_name,orca_args",
        [
            ("create_project", (), "create_project", ()),
            ("store_project", ("gated",), "update_project", ("gated",)),
            ("delete_project", ("gated",), "delete_project", ("gated",)),
        ],
    )
    def test_orca_direct_mode_delegates_and_skips_api_call(
        self, db, method_name, args, orca_method_name, orca_args
    ):
        mlrun.mlconf.httpdb.authentication.mode = "iguazio-v4"
        mlrun.mlconf.iguazio_api_url = ORCA_API_URL
        assert db._orca_direct_mode() is True

        fake_client = unittest.mock.Mock()
        db._orca_projects_client = fake_client

        project = _mlrun_project("gated")
        call_args = (
            (project,)
            if method_name in ("create_project",)
            else (
                *args,
                project,
            )
        )

        with unittest.mock.patch.object(db, "api_call") as mock_api_call:
            getattr(db, method_name)(*call_args)
            assert not mock_api_call.called

        # HTTPRunDB itself has no wait_for_completion knob (see ML-12903 discussion: no known
        # SDK caller needs async project creation, and exposing it would only work in this one
        # mode) - it always asks the Orca client to block until the operation settles.
        orca_method = getattr(fake_client, orca_method_name)
        orca_method.assert_called_once()
        assert orca_method.call_args.kwargs["wait_for_completion"] is True

    def test_orca_direct_mode_patch_project_coerces_patch_mode(self, db):
        mlrun.mlconf.httpdb.authentication.mode = "iguazio-v4"
        mlrun.mlconf.iguazio_api_url = ORCA_API_URL

        fake_client = unittest.mock.Mock()
        db._orca_projects_client = fake_client

        db.patch_project("gated", {"spec": {"description": "d"}}, patch_mode="additive")

        fake_client.patch_project.assert_called_once()
        assert (
            fake_client.patch_project.call_args.kwargs["patch_mode"]
            == mlrun.common.schemas.PatchMode.additive
        )

    def test_get_project_is_not_gated(self, db):
        # get_project is explicitly out of scope for this ticket (reads stay on the legacy
        # MLRun-API path) - it must not check _orca_direct_mode at all.
        mlrun.mlconf.httpdb.authentication.mode = "iguazio-v4"
        mlrun.mlconf.iguazio_api_url = ORCA_API_URL

        with unittest.mock.patch.object(db, "api_call") as mock_api_call:
            mock_api_call.return_value.json.return_value = _mlrun_project(
                "gated"
            ).to_dict()
            db.get_project("gated")
            assert mock_api_call.called
