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
import re
import unittest.mock
from contextlib import nullcontext as does_not_raise

import pytest
from kubernetes import client

from mlrun.api.utils.builder import (
    _parse_extra_args,
    make_dockerfile,
    make_kaniko_pod,
    validate_extra_args,
    _validate_and_merge_args_with_extra_args
)


@pytest.mark.parametrize(
    "builder_env,source,commands,project_secrets",
    [
        (
            [client.V1EnvVar(name="GIT_TOKEN", value="blakjhuy")],
            None,
            ["git+https://${GIT_TOKEN}@github.com/GiladShapira94/new-mlrun.git}"],
            {"SECRET": "secret"},
        ),
        (
            [client.V1EnvVar(name="GIT_TOKEN", value="blakjhuy")],
            "source.zip",
            ["echo bla"],
            {"SECRET": "secret", "ANOTHER": "secret"},
        ),
        (
            [
                client.V1EnvVar(name="GIT_TOKEN", value="jfksjnflsfnhg"),
                client.V1EnvVar(name="TEST", value="test"),
            ],
            "source.zip",
            [],
            {},
        ),
        (None, "source.zip", [], None),
    ],
)
def test_make_dockerfile_with_build_args(
    builder_env, source, commands, project_secrets
):
    dock = make_dockerfile(
        base_image="mlrun/mlrun",
        builder_env=builder_env,
        source=source,
        commands=commands,
        project_secrets=project_secrets,
    )
    print(dock)
    builder_env = builder_env or []
    project_secrets = project_secrets or {}
    # Check the resulted Dockerfile starts with the expected ARG lines
    expected_args = [f"ARG {env.name}={env.value}" for env in builder_env] + [
        f"ARG {key}={value}" for key, value in project_secrets.items()
    ]
    assert dock.startswith("\n".join(expected_args))

    # Check that the ARGS and ENV vars are declared in each stage of the Dockerfile
    expected_in_stage = (
        [f"ARG {env.name}" for env in builder_env]
        + [f"ARG {key}" for key in project_secrets]
        + [f"ENV {env.name}=${env.name}" for env in builder_env]
    )
    pattern = r"^FROM.*$"
    lines = dock.strip().split("\n")
    lines = [line.strip() for line in lines]

    for i, line in enumerate(lines):
        if re.match(pattern, line):
            assert lines[i + 1 : i + 1 + len(expected_in_stage)] == expected_in_stage


@pytest.mark.parametrize(
    "builder_env",
    [
        ([client.V1EnvVar(name="GIT_TOKEN", value="blakjhuy")]),
        (
            [
                client.V1EnvVar(name="GIT_TOKEN", value="blakjhuy"),
                client.V1EnvVar(name="TEST", value="test"),
            ]
        ),
        ([]),
    ],
)
def test_make_kaniko_pod_command_using_build_args(builder_env):
    with unittest.mock.patch(
        "mlrun.api.api.utils.resolve_project_default_service_account",
        return_value=(None, None),
    ):
        kpod = make_kaniko_pod(
            project="test",
            context="/context",
            dest="docker-hub/",
            dockerfile="./Dockerfile",
            builder_env=builder_env,
        )

    expected_env_vars = [f"{env_var.name}={env_var.value}" for env_var in builder_env]
    args = kpod.args
    actual_env_vars = [
        args[i + 1] for i in range(len(args)) if args[i] == "--build-arg"
    ]
    assert expected_env_vars == actual_env_vars


@pytest.mark.parametrize(
    "extra_args,expected_result",
    [
        ("--arg1 value1", {"--arg1": ["value1"]}),
        ("--arg1 value1 --arg2 value2", {"--arg1": ["value1"], "--arg2": ["value2"]}),
        ("--arg1 value1 value2 value3 --arg2 value4 value5",
         {"--arg1": ["value1", "value2", "value3"], "--arg2": ["value4", "value5"]}),
        ("--arg1 --arg2", {"--arg1": [], "--arg2": []}),
        ("--arg1 value1 --arg1 value2", {"--arg1": ["value1", "value2"]}),
        ("--arg1 value1 --arg2 value2 --arg1 value3", {"--arg1": ["value1", "value3"], "--arg2": ["value2"]}),
        ("", {}),
    ],
)
def test_parse_extra_args(extra_args, expected_result):
    assert _parse_extra_args(extra_args) == expected_result


@pytest.mark.parametrize(
    "extra_args,expected",
    [
        ("--build-arg KEY1=VALUE1 --build-arg KEY2=VALUE2", does_not_raise()),
        ("--build-arg KEY=VALUE", does_not_raise()),
        ("--build-arg KEY=VALUE key2=value2", does_not_raise()),
        ("--build-arg KEY=abc_ABC", does_not_raise()),
        (
            "--build-arg",
            pytest.raises(
                ValueError,
                match="Invalid '--build-arg' usage. It must be followed by a non-flag argument.",
            ),
        ),
        (
            "--build-arg KEY=VALUE invalid_argument",
            pytest.raises(
                ValueError,
                match="Invalid arguments format: 'invalid_argument'. Please make sure all arguments are in a valid format",
            ),
        ),
        (
            "a5 --build-arg --tls --build-arg a=7 c",
            pytest.raises(
                ValueError,
                match="Invalid argument sequence. Value must be followed by a flag preceding it.",
            ),
        ),
        (
            "--build-arg a=3 b=4 --tls --build-arg a=7 c d",
            pytest.raises(
                ValueError,
                match="Invalid arguments format: 'c,d'. Please make sure all arguments are in a valid format",
            ),
        ),
    ],
)
def test_validate_extra_args(extra_args, expected):
    with expected:
        validate_extra_args(extra_args)



@pytest.mark.parametrize(
    "args, extra_args, expected_result",
    [
        # Test cases with different input arguments and expected results
        (
            ["--arg1", "--arg2", "value2"],
            "--build-arg KEY1=VALUE1 --build-arg KEY2=VALUE2",
            ["--arg1", "--arg2", "value2", "--build-arg", "KEY1=VALUE1", "--build-arg", "KEY2=VALUE2"],
        ),
        (
            ["--arg1", "--arg2", "value2"],
            "--build-arg KEY1=VALUE1 --arg1 new_value1 --build-arg KEY2=new_value2",
            ['--arg1', '--arg2', 'value2', '--build-arg', 'KEY1=VALUE1', '--build-arg', 'KEY2=new_value2'],
        ),
        (
            ["--arg1", "value1"],
            "--build-arg KEY1=VALUE1 --build-arg KEY2=VALUE2",
            ["--arg1", "value1", "--build-arg", "KEY1=VALUE1", "--build-arg", "KEY2=VALUE2"],
        ),
        (
            ["--arg1", "--build-arg", "KEY1=VALUE1"],
            "--build-arg KEY2=VALUE2",
            ["--arg1", "--build-arg", "KEY1=VALUE1", "--build-arg", "KEY2=VALUE2"],
        ),
        (
            [],
            "--build-arg KEY1=VALUE1",
            ["--build-arg", "KEY1=VALUE1"],
        ),
        (
            ["--arg1"],
            "--build-arg KEY1=VALUE1",
            ["--arg1", "--build-arg", "KEY1=VALUE1"],
        ),
        (
            [],
            "",
            [],
        ),
    ],
)
def test_validate_and_merge_args_with_extra_args(args, extra_args, expected_result):
    assert _validate_and_merge_args_with_extra_args(args, extra_args) == expected_result