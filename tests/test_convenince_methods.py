# Copyright 2018 Iguazio
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
import subprocess
import sys

import dotenv
import pytest

import mlrun
import mlrun.projects.project
from tests.conftest import out_path

assets_path = pathlib.Path(__file__).parent / "assets"


def test_set_environment_with_invalid_project_name():
    invalid_name = "project_name"
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mlrun.set_environment(project=invalid_name)


def test_set_environment_cred():
    old_key = os.environ.get("V3IO_ACCESS_KEY")
    old_user = os.environ.get("V3IO_USERNAME")
    artifact_path = mlrun.mlconf.artifact_path or "./"
    mlrun.set_environment(access_key="xyz", username="joe", artifact_path=artifact_path)
    assert os.environ["V3IO_ACCESS_KEY"] == "xyz"
    assert os.environ["V3IO_USERNAME"] == "joe"

    if old_key:
        os.environ["V3IO_ACCESS_KEY"] = old_key
    else:
        del os.environ["V3IO_ACCESS_KEY"]
    if old_user:
        os.environ["V3IO_USERNAME"] = old_user
    else:
        del os.environ["V3IO_USERNAME"]


def test_env_from_file():
    env_path = str(assets_path / "envfile")
    env_dict = mlrun.set_env_from_file(env_path, return_dict=True)
    assert env_dict == {"ENV_ARG1": "123", "ENV_ARG2": "abc", "MLRUN_KFP_TTL": "12345"}
    assert mlrun.mlconf.kfp_ttl == 12345
    for key, value in env_dict.items():
        assert os.environ[key] == value
    for key in env_dict.keys():
        del os.environ[key]

    # test setting env_file using set_environment()
    artifact_path = mlrun.mlconf.artifact_path or "./"
    mlrun.set_environment(env_file=env_path, artifact_path=artifact_path)
    for key, value in env_dict.items():
        assert os.environ[key] == value
    for key in env_dict.keys():
        del os.environ[key]


def test_mock_functions():
    mock_nuclio_config = mlrun.mlconf.mock_nuclio_deployment
    local_config = mlrun.mlconf.force_run_local

    # test setting env_file using set_environment()
    artifact_path = mlrun.mlconf.artifact_path or "./"
    mlrun.set_environment(mock_functions=True, artifact_path=artifact_path)
    assert mlrun.mlconf.mock_nuclio_deployment == "1"
    assert mlrun.mlconf.force_run_local == "1"

    mlrun.set_environment(mock_functions="auto", artifact_path=artifact_path)
    assert mlrun.mlconf.mock_nuclio_deployment == "auto"
    assert mlrun.mlconf.force_run_local == "auto"

    mlrun.mlconf.mock_nuclio_deployment = mock_nuclio_config
    mlrun.mlconf.force_run_local = local_config


def test_bad_env_files():
    bad_envs = ["badenv1"]

    for env in bad_envs:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            mlrun.set_env_from_file(str(assets_path / env))


def test_env_file_does_not_exist():
    with pytest.raises(mlrun.errors.MLRunNotFoundError):
        mlrun.set_env_from_file("some-nonexistent-path")


def test_auto_load_env_file():
    os.environ["MLRUN_ENV_FILE"] = str(assets_path / "envfile")
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


def test_set_config():
    env_path = f"{out_path}/env/myenv.env"
    api = "http://localhost:8080"
    pathlib.Path(env_path).parent.mkdir(parents=True, exist_ok=True)
    _exec_mlrun(
        f"config set -f {env_path} -a {api} -u joe -k mykey -p /c/y -e XXX=myvar"
    )

    expected = {
        "MLRUN_DBPATH": api,
        "MLRUN_ARTIFACT_PATH": "/c/y",
        "V3IO_USERNAME": "joe",
        "V3IO_ACCESS_KEY": "mykey",
        "XXX": "myvar",
    }
    env_vars = dotenv.dotenv_values(env_path)
    assert len(env_vars) == 5
    for key, val in expected.items():
        assert env_vars[key] == val


def test_set_and_load_default_config():
    env_path = os.path.expanduser(mlrun.config.default_env_file)
    env_body = None
    if os.path.isfile(env_path):
        with open(env_path, "r") as fp:
            env_body = fp.read()

    # set two config (mlrun and custom vars) and read/verify the default .env file
    _exec_mlrun("config set -e YYYY=myvar -e MLRUN_KFP_TTL=12345")
    env_vars = dotenv.dotenv_values(env_path)
    assert env_vars["YYYY"] == "myvar"
    assert env_vars["MLRUN_KFP_TTL"] == "12345"

    # verify the new env impact mlrun config
    mlrun.mlconf.reload()
    assert mlrun.mlconf.kfp_ttl == 12345

    _exec_mlrun("config clear")
    assert not os.path.isfile(env_path), "config file was not deleted"

    # write back old content and del env vars
    if env_body:
        with open(env_path, "w") as fp:
            fp.write(env_body)
    print(os.environ)
    if "YYYY" in os.environ:
        del os.environ["YYYY"]
    if "MLRUN_KFP_TTL" in os.environ:
        del os.environ["MLRUN_KFP_TTL"]


def _exec_mlrun(cmd, cwd=None):
    cmd = [sys.executable, "-m", "mlrun"] + cmd.split()
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    if out.returncode != 0:
        print(out.stderr.decode("utf-8"), file=sys.stderr)
        print(out.stdout.decode("utf-8"), file=sys.stderr)
        raise Exception(out.stderr.decode("utf-8"))
    return out.stdout.decode("utf-8")
