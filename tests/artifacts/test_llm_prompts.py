# Copyright 2025 Iguazio
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

import pytest

import mlrun
import mlrun.artifacts
from tests import conftest

results_dir = (pathlib.Path(conftest.results) / "artifacts").absolute()
llm_file = pathlib.Path(__file__).parent / "assets" / "prompt.txt"


@pytest.mark.parametrize(
    "generate_target_path_from_artifact_hash",
    [True, False],
)
@pytest.mark.parametrize(
    "from_file",
    [True, False],
)
def test_prompt_target_paths(generate_target_path_from_artifact_hash, from_file):
    mlrun.mlconf.artifacts.generate_target_path_from_artifact_hash = (
        generate_target_path_from_artifact_hash
    )
    project_name = "project-test"
    artifact_path = results_dir / project_name
    llm_key = "llm-prompt"

    context = mlrun.get_or_create_ctx("test", project=project_name)
    # we use log artifact and not log model as it should handle models as well
    if from_file:
        llm_prompt = context.log_llm_prompt(
            llm_key,
            artifact_path=artifact_path,
            prompt_path=str(llm_file),
            description="best-prompt",
        )
    else:
        llm_prompt = context.log_llm_prompt(
            llm_key,
            artifact_path=artifact_path,
            prompt_string="Q : {question}",
            description="best-prompt",
        )
    assert llm_prompt.target_path.startswith(str(artifact_path))

    prompt_template = llm_prompt.read_prompt()
    assert prompt_template == "Q : {question}"


def test_prompt_limitation():
    project_name = "project-test"
    artifact_path = results_dir / project_name
    llm_key = "llm-prompt"

    context = mlrun.get_or_create_ctx("test", project=project_name)
    # we use log artifact and not log model as it should handle models as well

    llm_prompt = context.log_llm_prompt(
        llm_key,
        artifact_path=artifact_path,
        prompt_string="A" * 2000,
        description="long-prompt",
    )
    assert llm_prompt.target_path.startswith(str(artifact_path))
    assert llm_prompt.spec.prompt_string is None

    prompt_template = llm_prompt.read_prompt()
    assert prompt_template == "A" * 2000
