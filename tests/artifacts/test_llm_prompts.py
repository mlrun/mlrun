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
llm_file = pathlib.Path(__file__).parent / "assets" / "prompt.json"


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
    artifact_path = str(results_dir / project_name)
    llm_key = "llm-prompt"

    context = mlrun.get_or_create_ctx("test", project=project_name)
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
            prompt_template=[
                {"role": "system", "content": "remarks"},
                {"role": "user", "content": "question"},
            ],
            description="best-prompt",
        )
    assert llm_prompt.target_path.startswith(str(artifact_path))

    prompt_template = llm_prompt.read_prompt()
    assert prompt_template == [
        {"role": "system", "content": "remarks"},
        {"role": "user", "content": "question"},
    ]


def test_prompt_limitation():
    project_name = "project-test"
    artifact_path = str(results_dir / project_name)
    llm_key = "llm-prompt"

    context = mlrun.get_or_create_ctx("test", project=project_name)

    llm_prompt = context.log_llm_prompt(
        llm_key,
        artifact_path=artifact_path,
        prompt_template=[{"role": "user", "content": "A" * 2000}],
        description="long-prompt",
    )
    assert llm_prompt.target_path.startswith(str(artifact_path))
    assert llm_prompt.spec.prompt_template is None

    prompt_template = llm_prompt.read_prompt()
    assert prompt_template == [{"role": "user", "content": "A" * 2000}]
    assert llm_prompt.target_path.endswith(f"{llm_key}.json")


@pytest.mark.parametrize(
    "prompt_template",
    [
        [{"role": "user", "content": "A", "should_not_be_prompted": "I am here"}],
        "just a regular str",
        {"role": "user", "content": "A"},
        [{"role": "user"}],
        [{"content": "A"}],
    ],
)
def test_prompt_template_verification(prompt_template):
    project_name = "project-test"
    artifact_path = str(results_dir / project_name)
    llm_key = "llm-prompt"

    context = mlrun.get_or_create_ctx("test", project=project_name)
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        context.log_llm_prompt(
            llm_key,
            artifact_path=artifact_path,
            prompt_template=prompt_template,
            description="long-prompt",
        )


@pytest.mark.parametrize(
    "prompt_legend ,with_failure",
    [
        (
            {
                "country": {
                    "field": "my_country",
                    "description": "my-country-description",
                }
            },
            False,
        ),
        (
            {
                "country": {
                    "field": "my_country",
                    "description": "my-country-description",
                    "another_field": "not here",
                }
            },
            True,
        ),
    ],
)
def test_prompt_legend(prompt_legend, with_failure):
    project_name = "project-test"
    artifact_path = str(results_dir / project_name)
    llm_key = "llm-prompt"

    context = mlrun.get_or_create_ctx("test", project=project_name)
    if with_failure:
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            context.log_llm_prompt(
                llm_key,
                artifact_path=artifact_path,
                prompt_template=[{"role": "user", "content": "A {country}"}],
                description="long-prompt",
                prompt_legend=prompt_legend,
            )
    else:
        context.log_llm_prompt(
            llm_key,
            artifact_path=artifact_path,
            prompt_template=[{"role": "user", "content": "A {country}"}],
            description="long-prompt",
            prompt_legend=prompt_legend,
        )


@pytest.mark.parametrize(
    "project_name_llm",
    ["project-test-1", None],
)
def test_unauthorised_model(project_name_llm):
    project_name_model = "project-test"
    artifact_path_llm = str(results_dir / (project_name_llm or ""))
    artifact_path_model = str(results_dir / project_name_model)
    llm_key = "llm-prompt"
    model_key = "model"

    context_model = mlrun.get_or_create_ctx("test", project=project_name_model)

    model = context_model.log_model(
        model_key,
        body="model body",
        model_file="trained_model.pkl",
        artifact_path=artifact_path_model,
    )

    context_llm = mlrun.get_or_create_ctx("test-1", project=project_name_llm)
    context_llm._project = project_name_llm
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        # different projects
        context_llm.log_llm_prompt(
            llm_key,
            artifact_path=artifact_path_llm,
            prompt_template=[{"role": "user", "content": "A" * 2000}],
            description="long-prompt",
            model_artifact=model,
        )

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        # bad model uri
        context_llm.log_llm_prompt(
            llm_key,
            artifact_path=artifact_path_llm,
            prompt_template=[{"role": "user", "content": "A" * 2000}],
            description="long-prompt",
            model_artifact="dasdcfsfv",
        )
