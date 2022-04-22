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


import re
from collections import ChainMap
from os import environ
from pathlib import Path
from subprocess import run

import pytest
import yaml

here = Path(__file__).absolute().parent
tests_dir = here.parent
root = tests_dir.parent
# Need to be in root for docker context
tmp_dockerfile = Path(root / "Dockerfile.mlrun-test-nb")
with (here / "Dockerfile.test-nb").open() as fp:
    dockerfile_template = fp.read()
docker_tag = "mlrun/test-notebook"


def mlrun_api_configured():
    config_file_path = here / "test-notebooks.yml"
    with config_file_path.open() as fp:
        config = yaml.safe_load(fp)
    return config["env"].get("MLRUN_DBPATH") is not None


def iterate_notebooks():
    if not mlrun_api_configured():
        return []
    config_file_path = here / "test-notebooks.yml"
    with config_file_path.open() as fp:
        config = yaml.safe_load(fp)

    general_env = config["env"]

    for notebook_test_config in config["notebook_tests"]:

        # fill env keys that reference the general env
        test_env = {}
        for key, value in notebook_test_config.get("env", {}).items():
            match = re.match(r"^\$\{(?P<env_var>.*)\}$", value)
            if match is not None:
                env_var = match.group("env_var")
                env_var_value = general_env.get(env_var)
                if env_var_value is None:
                    raise ValueError(
                        f"Env var {env_var} references general env, but it does not exist there"
                    )
                test_env[key] = env_var_value
            else:
                test_env[key] = value
        notebook_test_config["env"] = test_env

        yield pytest.param(
            notebook_test_config, id=notebook_test_config["notebook_name"]
        )


def args_from_env(env):
    external_env = {}
    for env_var_key in environ:
        if env_var_key.startswith("MLRUN_"):
            external_env[env_var_key] = environ[env_var_key]
    env = ChainMap(env, external_env)
    args, cmd = [], []
    for name in env:
        value = env[name]
        args.append(f"ARG {name}")
        cmd.extend(["--build-arg", f"{name}={value}"])

    args = "\n".join(args)
    return args, cmd


@pytest.mark.skipif(
    not mlrun_api_configured(),
    reason="This is an integration test, add the needed environment variables in test-notebooks.yml "
    "to run it",
)
@pytest.mark.parametrize("notebook", iterate_notebooks())
def test_notebook(notebook):
    path = f'./examples/{notebook["notebook_name"]}'
    args, args_cmd = args_from_env(notebook["env"])
    deps = []
    for dep in notebook.get("pip", []):
        deps.append(f"RUN python -m pip install --upgrade {dep}")
    pip = "\n".join(deps)

    code = dockerfile_template.format(notebook=path, args=args, pip=pip)
    with tmp_dockerfile.open("w") as out:
        out.write(code)

    cmd = (
        ["docker", "build", "--file", str(tmp_dockerfile), "--tag", docker_tag]
        + args_cmd
        + ["."]
    )
    out = run(cmd, cwd=root)
    assert out.returncode == 0, "cannot build"
