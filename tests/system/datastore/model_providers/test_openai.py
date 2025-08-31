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
import json
import os
import time

import pytest
import tiktoken

from mlrun.datastore.datastore_profile import (
    OpenAIProfile,
)
from mlrun.datastore.model_provider.model_provider import UsageResponseKeys
from tests.datastore.remote_model.remote_model_utils import (
    EXPECTED_RESULTS,
    INPUT_DATA,
    assert_async_invocations,
    setup_remote_model_test,
)
from tests.system.base import TestMLRunSystem


def get_missing_openai_env_variables():
    return [
        env_key
        for env_key in ["OPENAI_BASE_URL", "OPENAI_API_KEY"]
        if not os.environ.get(env_key)
    ]


@TestMLRunSystem.skip_test_if_env_not_configured
class TestOpenAIModelRunner(TestMLRunSystem):
    """Applying basic model endpoint CRUD operations through MLRun API"""

    project_name = "openai-system-test"
    image = "mlrun/mlrun"
    profile_name = "my_openai_profile"

    @classmethod
    def setup_class(cls):
        super().setup_class()
        missing_env_variables = get_missing_openai_env_variables()
        if missing_env_variables:
            pytest.skip(
                f"The following openai keys are missing: {missing_env_variables}"
            )
        cls.basic_llm_model = "gpt-4o-mini"

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        self.profile = OpenAIProfile(
            name=self.profile_name,
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORG_ID"),
            project=os.environ.get("OPENAI_PROJECT_ID"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            timeout=os.environ.get("OPENAI_TIMEOUT"),
            max_retries=os.environ.get("OPENAI_MAX_RETRIES"),
        )
        self.project.register_datastore_profile(self.profile)
        self.url_prefix = f"ds://{self.profile_name}/"
        self.model_url = self.url_prefix + self.basic_llm_model

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_basic_openai_model_runner(self, execution_mechanism):
        mlrun_model_name = "sync_invoke_model"
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            self.model_url,
            mlrun_model_name=mlrun_model_name,
            image=self.image,
            requirements=["openai==1.77.0"],
            execution_mechanism=execution_mechanism,
            default_config={"max_tokens": 100},
        )
        function.deploy()
        response = function.invoke(
            f"v2/models/{mlrun_model_name}/infer",
            json.dumps(INPUT_DATA[0]),
        )["output"]
        assert len(response) == 2
        answer = response[UsageResponseKeys.ANSWER]
        assert EXPECTED_RESULTS[0] in answer.lower()
        encoding = tiktoken.encoding_for_model(self.basic_llm_model)
        assert len(encoding.encode(answer)) == 100

        stats = response[UsageResponseKeys.USAGE]
        assert stats["completion_tokens"] == 100
        assert stats["prompt_tokens"] > 0
        assert (
            stats["total_tokens"] == stats["completion_tokens"] + stats["prompt_tokens"]
        )

    def test_model_runner_with_openai_async(self):
        mlrun_model_name = "async_invoke_model"
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            self.model_url,
            mlrun_model_name=mlrun_model_name,
            execution_mechanism="asyncio",
            image=self.image,
            requirements=["openai==1.77.0"],
            model_class="MyOpenAIAsyncEvents",
            default_config={"max_tokens": 100},
        )
        function.deploy()

        start = time.perf_counter()
        results_with_times = function.invoke(
            f"v2/models/{mlrun_model_name}/infer",
            json.dumps({"input": INPUT_DATA}),
        )
        total_duration = time.perf_counter() - start
        assert_async_invocations(
            results_with_times=results_with_times,
            model_name=self.basic_llm_model,
            total_duration=total_duration,
        )
