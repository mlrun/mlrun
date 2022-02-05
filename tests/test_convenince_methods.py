import os
import pathlib

import pytest

import mlrun
import mlrun.projects.project

assets_path = pathlib.Path(__file__).parent / "assets"


def test_set_environment_with_invalid_project_name():
    invalid_name = "project_name"
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.set_environment(project=invalid_name)


def test_env_from_file():
    env_path = str(assets_path / "envfile")
    env_dict = mlrun.set_env_from_file(env_path, return_dict=True)
    assert env_dict == {"ENV_ARG1": "123", "ENV_ARG2": "abc", "MLRUN_KFP_TTL": "12345"}
    assert mlrun.mlconf.kfp_ttl == 12345
    for key, value in env_dict.items():
        assert os.environ[key] == value
    for key in env_dict.keys():
        del os.environ[key]


def test_bad_env_files():
    bad_envs = ["badenv1"]

    for env in bad_envs:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            mlrun.set_env_from_file(str(assets_path / env))


def test_auto_load_env_file():
    os.environ["MLRUN_SET_ENV_FILE"] = str(assets_path / "envfile")
    mlrun.mlconf.reload()
    assert mlrun.mlconf.kfp_ttl == 12345
    expected = {"ENV_ARG1": "123", "ENV_ARG2": "abc", "MLRUN_KFP_TTL": "12345"}
    count = 0
    for key in os.environ.keys():
        if key in expected:
            assert os.environ[key] == expected[key]
            count += 1
    assert count == 3
    for key in expected.keys():
        del os.environ[key]


def test_deduct_v3io_paths():
    cluster = ".default-tenant.app.xxx.iguazio-cd1.com"
    conf = mlrun.config.read_env({"MLRUN_DBPATH": "https://mlrun-api" + cluster})
    assert conf["v3io_api"] == "https://webapi" + cluster
    assert conf["v3io_framesd"] == "https://framesd" + cluster
