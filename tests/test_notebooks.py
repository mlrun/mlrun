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


from collections import ChainMap
from os import environ
from pathlib import Path
from subprocess import run

import pytest
import yaml

here = Path(__file__).absolute().parent
root = here.parent
# Need to be in root for docker context
tmp_dockerfile = Path(root / "Dockerfile.mlrun-test-nb")
with (here / "Dockerfile.test-nb").open() as fp:
    dockerfile_template = fp.read()
docker_tag = "mlrun/test-notebook"


def iter_notebooks():
    cfg_file = here / "notebooks.yml"
    with cfg_file.open() as fp:
        configs = yaml.safe_load(fp)

    for config in configs:
        if "env" not in config:
            config["env"] = {}
        yield pytest.param(config, id=config["nb"])


def args_from_env(env):
    env = ChainMap(env, environ)
    args, cmd = [], []
    for name in env:
        if not name.startswith("MLRUN_"):
            continue
        value = env[name]
        args.append(f"ARG {name}")
        cmd.extend(["--build-arg", f"{name}={value}"])

    args = "\n".join(args)
    return args, cmd


@pytest.mark.parametrize("notebook", iter_notebooks())
def test_notebook(notebook):
    path = f'./examples/{notebook["nb"]}'
    args, args_cmd = args_from_env(notebook["env"])
    deps = []
    for dep in notebook.get("pip", []):
        deps.append(f"RUN python -m pip install {dep}")
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
    assert out.returncode == 0, f"Failed building {out.stdout} {out.stderr}"
