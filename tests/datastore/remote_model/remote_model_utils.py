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
import mlrun
from mlrun.serving import ModelRunnerStep

INPUT_DATA = {
    "input": [
        {
            "question": "What is the capital of France, and give a brief historical overview.",
            "depth_level": "detailed",
            "persona": "teacher",
            "tone": "casual",
        },
        {
            "question": "What is 2 + 2? Answer shortly and then explain with details.",
            "depth_level": "basic",
            "persona": "math teacher",
            "tone": "simple",
        },
        {
            "question": "Who wrote Hamlet? Answer shortly and then explain with details.",
            "depth_level": "basic",
            "persona": "literature professor",
            "tone": "formal",
        },
        {
            "question": "What color is the sky on a clear day? Answer shortly and then explain with details.",
            "depth_level": "basic",
            "persona": "child",
            "tone": "fun",
        },
        {
            "question": "What planet do we live on? Answer shortly and then explain with details.",
            "depth_level": "basic",
            "persona": "astronaut",
            "tone": "educational",
        },
    ],
}

EXPECTED_RESULTS = ["paris", "4", "shakespeare", "blue", "earth"]


def setup_remote_model_test(project, model_url):
    model_artifact = project.log_model(
        "my_model",
        model_url=model_url,
        default_config={"max_tokens": 100},
    )
    prompt_template = (
        "{question}. Explain {depth_level} as a {persona} in {tone} style."
    )
    llm_prompt_artifact = project.log_llm_prompt(
        "my_llm_prompt",
        prompt_string=prompt_template,
        model_artifact=model_artifact.uri,
    )
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    return model_artifact, llm_prompt_artifact, function, graph, model_runner_step
