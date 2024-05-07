# Copyright 2023 Iguazio
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
import pathlib

import mlrun.projects
from mlrun.__main__ import load_notification
from mlrun.artifacts.plots import PlotArtifact
from mlrun.lists import ArtifactList


def test_add_notification_to_cli_from_file():
    input_file_path = str(pathlib.Path(__file__).parent / "assets/notification.json")
    notifications = (f"file={input_file_path}",)
    project = mlrun.projects.MlrunProject(
        metadata=mlrun.projects.ProjectMetadata(name="test")
    )
    load_notification(notifications, project)

    assert (
        project._notifiers._async_notifications["slack"].params.get("webhook")
        == "123456"
    )
    assert (
        project._notifiers._sync_notifications["ipython"].params.get("webhook")
        == "1234"
    )


def test_add_notification_to_cli_from_dict():
    notifications = ('{"slack":{"webhook":"123456"}}', '{"ipython":{"webhook":"1234"}}')
    project = mlrun.projects.MlrunProject(
        metadata=mlrun.projects.ProjectMetadata(name="test")
    )
    load_notification(notifications, project)

    assert (
        project._notifiers._async_notifications["slack"].params.get("webhook")
        == "123456"
    )
    assert (
        project._notifiers._sync_notifications["ipython"].params.get("webhook")
        == "1234"
    )


def test_cli_get_artifacts_with_uri():
    artifacts = []
    for i in range(5):
        artifact_key = f"artifact_test_{i}"
        artifact_uid = f"artifact_uid_{i}"
        artifact_kind = PlotArtifact.kind
        artifacts.append(
            generate_artifact(artifact_key, kind=artifact_kind, uid=artifact_uid)
        )
    artifacts = ArtifactList(artifacts)

    # this is the function called when executing the get artifacts cli command
    df = artifacts.to_df()

    # check that the uri is returned
    assert "uri" in df


def generate_artifact(name, uid=None, kind=None):
    artifact = {
        "metadata": {"key": name, "iter": 0},
        "spec": {"src_path": "/some/path"},
        "kind": kind,
        "status": {"bla": "blabla"},
    }
    if kind:
        artifact["kind"] = kind
    if uid:
        artifact["metadata"]["uid"] = uid

    return artifact
