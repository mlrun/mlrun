import os
import pathlib

import pytest

import mlrun
import mlrun.errors
import mlrun.projects.project

assets_path = pathlib.Path(__file__).parent / "assets"


def test_set_environment_with_invalid_project_name():
    invalid_name = "project_name"
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.set_environment(project=invalid_name)


def test_env_from_file():
    env_path = str(assets_path / "envfile")
    env_dict = mlrun.env_from_file(env_path, to_dict=True)
    assert env_dict == {"ENV_ARG1": "123", "ENV_ARG2": "abc"}
    for key, value in env_dict.items():
        assert os.environ[key] == value
