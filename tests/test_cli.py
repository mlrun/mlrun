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
import unittest.mock

import pytest
from click.testing import CliRunner

import mlrun.projects
from mlrun.__main__ import load_notification, main
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


@pytest.mark.parametrize(
    "project",
    [None, "my-project"],
)
def test_cli_load_source_success(project):
    # Test load-source CLI with and without an explicit project
    runner = CliRunner()
    source_uri = "store://artifacts/my-project/handler.py"

    cli_args = ["load-source", source_uri]
    if project:
        cli_args.extend(["--project", project])

    with unittest.mock.patch(
        "mlrun.__main__.load_source_code",
        return_value=("/home/mlrun_code", "/home/mlrun_code/handler.py"),
    ) as mock_load:
        result = runner.invoke(main, cli_args)

    assert result.exit_code == 0
    assert "Successfully loaded source to:" in result.output
    mock_load.assert_called_once_with(
        source_uri=source_uri,
        target_dir="/home/mlrun_code",
        project=project,
    )


def test_cli_load_source_failure():
    # Test that CLI properly reports errors and exits with code 1
    runner = CliRunner()

    with unittest.mock.patch(
        "mlrun.__main__.load_source_code",
        side_effect=ValueError("Artifact not found"),
    ):
        result = runner.invoke(
            main,
            ["load-source", "store://artifacts/project/file.py"],
        )

    assert result.exit_code == 1
    assert "Error loading source:" in result.output


@pytest.mark.parametrize(
    "source_uri,expected_target",
    [
        ("git://github.com/org/repo.git#main", "/home/mlrun_code"),
        ("https://example.com/source.zip", "/home/mlrun_code"),
        ("https://example.com/source.tar.gz", "/home/mlrun_code"),
    ],
)
def test_cli_load_source_git_and_archives(source_uri, expected_target):
    # Test load-source CLI with git and archive sources
    runner = CliRunner()

    with unittest.mock.patch(
        "mlrun.__main__.load_source_code",
        return_value=(expected_target, None),
    ) as mock_load:
        result = runner.invoke(main, ["load-source", source_uri])

    assert result.exit_code == 0
    assert "Successfully loaded source to:" in result.output
    mock_load.assert_called_once_with(
        source_uri=source_uri,
        target_dir=expected_target,
        project=None,
    )


def test_cli_load_source_custom_target():
    # Test load-source CLI with custom target directory
    runner = CliRunner()
    source_uri = "git://github.com/org/repo.git"
    custom_target = "/custom/path"

    with unittest.mock.patch(
        "mlrun.__main__.load_source_code",
        return_value=(custom_target, None),
    ) as mock_load:
        result = runner.invoke(
            main, ["load-source", source_uri, "--target", custom_target]
        )

    assert result.exit_code == 0
    mock_load.assert_called_once_with(
        source_uri=source_uri,
        target_dir=custom_target,
        project=None,
    )
