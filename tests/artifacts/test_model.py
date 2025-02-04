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
#
import os
import pathlib
import uuid

import pytest
import yaml

import mlrun
import mlrun.artifacts
from tests import conftest

results_dir = (pathlib.Path(conftest.results) / "artifacts").absolute()
model_file = pathlib.Path(__file__).parent / "assets" / "model.pkl"


@pytest.mark.parametrize(
    "generate_target_path_from_artifact_hash, expected_model_target_file",
    [(True, "da39a3ee5e6b4b0d3255bfef95601890afd80709"), (False, None)],
)
def test_model_target_paths(
    generate_target_path_from_artifact_hash, expected_model_target_file
):
    mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash = (
        generate_target_path_from_artifact_hash
    )
    project_name = "model-target-path-test"
    artifact_path = results_dir / project_name
    model_key = "model"
    model = mlrun.artifacts.ModelArtifact(key=model_key, model_file=model_file)

    context = mlrun.get_or_create_ctx("test")
    # we use log artifact and not log model as it should handle models as well
    model = context.log_artifact(model, artifact_path=artifact_path)

    assert model.target_path.startswith(str(artifact_path))
    assert model.model_file == os.path.basename(model_file)
    assert model.model_target_file == expected_model_target_file


def test_tag_not_in_model_spec():
    model_name = "my-model"
    tag = "some-tag"
    project_name = "model-spec-test"
    artifact_path = results_dir / project_name

    # create a project and log a model
    project = mlrun.new_project(project_name, save=False)
    project.log_model(
        model_name,
        body="model body",
        model_file="trained_model.pkl",
        tag=tag,
        artifact_path=artifact_path,
        upload=True,
    )

    # list the artifact path dir and verify the model spec file exists
    model_path = artifact_path / model_name
    files = os.listdir(model_path)
    assert mlrun.artifacts.model.model_spec_filename in files

    # open the model spec file and verify the tag is not there
    with open(model_path / mlrun.artifacts.model.model_spec_filename) as f:
        model_spec = yaml.load(f, Loader=yaml.FullLoader)

    assert "tag" not in model_spec, "tag should not be in model spec"
    assert "tag" not in model_spec["metadata"], "tag should not be in metadata"


def test_sanitize_model_spec():
    model_key = "model_key"
    some_extra_data = "some_extra_data"
    future_extra_data = "future_extra_data"
    model = mlrun.artifacts.ModelArtifact(
        key=model_key,
        model_file=model_file,
        extra_data={
            some_extra_data: "abc",
            future_extra_data: ...,
        },
    )
    model.metadata.tag = "v1"
    sanitized_model_spec = mlrun.artifacts.model._sanitize_model_spec(model)
    assert "tag" not in sanitized_model_spec["metadata"]
    assert some_extra_data in sanitized_model_spec["spec"]["extra_data"]
    assert future_extra_data not in sanitized_model_spec["spec"]["extra_data"]


def test_get_model_without_suffix():
    project = mlrun.new_project("get-model-without-suffix")
    model_name = f"custom_suffix_model_{uuid.uuid4()}"
    for suffix in mlrun.artifacts.model.MODEL_OPTIONAL_SUFFIXES:
        file_name = f"model.{suffix}"
        project.log_model(
            model_name,
            body=b"123",
            model_file=file_name,
            artifact_path=results_dir,
        )
        model_dir_path = os.path.join(results_dir, model_name)
        model_path = os.path.join(model_dir_path, file_name)
        temp_path, _, _ = mlrun.artifacts.get_model(model_path)
        assert temp_path == model_path
        with open(temp_path, "rb") as fp:
            data = fp.read()
        assert data == b"123"


def test_get_model_with_dataitem(rundb_mock, tmpdir: pathlib.Path):
    model_name = "my-model"
    tag = "some-tag"
    project_name = "get-model-with-data-item"
    artifact_path = tmpdir / project_name
    body = "model body"
    file_name = "trained_model.pkl"

    # create a project and log a model
    project = mlrun.new_project(project_name, save=False)
    model_artifact = project.log_model(
        model_name,
        body=body,
        model_file=file_name,
        tag=tag,
        artifact_path=artifact_path,
        upload=True,
    )

    model_dataitem = model_artifact.to_dataitem()
    model_path, model_spec, _ = mlrun.artifacts.get_model(model_dataitem)
    assert f"{artifact_path}/{model_name}/{file_name}" == model_path
