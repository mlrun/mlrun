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
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
import tiktoken

import mlrun
from mlrun.datastore.datastore_profile import (
    OpenAIProfile,
)
from mlrun.runtimes.nuclio.function import AsyncSpec
from tests.datastore.remote_model.remote_model_utils import (
    BATCH_INPUT_DATA,
    EXPECTED_RESULTS,
    assert_async_invocations,
    retry_on_content_mismatch,
    setup_remote_model_test,
    validate_openai_batch_response,
    validate_openai_single_response,
)
from tests.system.base import TestMLRunSystem

MAX_ATTEMPTS = 3


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
        cls.embedding_model = "text-embedding-3-small"

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

        def _test_single():
            response = function.invoke(
                f"v2/models/{mlrun_model_name}/infer",
                json.dumps(BATCH_INPUT_DATA[0]),
            )["output"]
            validate_openai_single_response(
                response, EXPECTED_RESULTS[0], self.basic_llm_model
            )

        retry_on_content_mismatch(_test_single, MAX_ATTEMPTS)

        def _test_batch():
            batch_response = function.invoke(
                f"v2/models/{mlrun_model_name}/infer",
                json.dumps(BATCH_INPUT_DATA),
            )
            validate_openai_batch_response(
                batch_response, EXPECTED_RESULTS, self.basic_llm_model
            )

        retry_on_content_mismatch(_test_batch, MAX_ATTEMPTS)

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_openai_model_runner_batch_step(self, execution_mechanism):
        mlrun_model_name = "batch_step_model"
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            self.model_url,
            mlrun_model_name=mlrun_model_name,
            image=self.image,
            requirements=["openai==1.77.0"],
            execution_mechanism=execution_mechanism,
            default_config={"max_tokens": 100},
            batch_step=True,
        )
        function.with_http(workers=None, async_spec=AsyncSpec())
        function.deploy()

        # Send events concurrently with staggered timing
        def send_event(event, delay):
            time.sleep(delay)
            return function.invoke(
                f"v2/models/{mlrun_model_name}/infer",
                event,
            )

        def _test():
            with ThreadPoolExecutor(max_workers=len(BATCH_INPUT_DATA)) as executor:
                futures = [
                    executor.submit(send_event, event, i * 0.1)
                    for i, event in enumerate(BATCH_INPUT_DATA)
                ]
                batch_response = [future.result() for future in futures]
            validate_openai_batch_response(
                batch_response, EXPECTED_RESULTS, self.basic_llm_model
            )

        retry_on_content_mismatch(_test, MAX_ATTEMPTS)

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

        for attempt in range(1, MAX_ATTEMPTS + 1):
            start = time.perf_counter()
            results_with_times = function.invoke(
                f"v2/models/{mlrun_model_name}/infer",
                json.dumps({"input": BATCH_INPUT_DATA}),
            )
            total_duration = time.perf_counter() - start

            try:
                assert_async_invocations(
                    results_with_times=results_with_times,
                    model_name=self.basic_llm_model,
                    total_duration=total_duration,
                )
                # success, break out of the retry loop
                break
            except AssertionError as e:
                if attempt < MAX_ATTEMPTS:
                    print(f"[Attempt {attempt}] Assertion failed, retrying...")
                    continue
                else:
                    print(f"[Attempt {attempt}] Giving up after {MAX_ATTEMPTS} tries.")
                    raise e

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_open_ai_custom(self, execution_mechanism):
        mlrun_model_name = "custom_invoke_model"
        model_url = self.url_prefix + self.embedding_model
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            model_url,
            mlrun_model_name=mlrun_model_name,
            execution_mechanism=execution_mechanism,
            image=self.image,
            requirements=["openai==1.77.0"],
            model_class="MyOpenAICustom",
            default_config={"dimensions": 256},
        )
        function.deploy()
        prompt = "Hello GPT"
        result = function.invoke(
            f"v2/models/{mlrun_model_name}/infer",
            json.dumps({"input": prompt}),
        )["result"]
        encoding = tiktoken.encoding_for_model(self.embedding_model)
        token_count = len(encoding.encode(prompt))
        assert len(result["data"][0]["embedding"]) == 256
        assert result["usage"]["total_tokens"] == token_count

    def test_openai_model_runner_streaming(self):
        mlrun_model_name = "streaming_model"
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            self.model_url,
            mlrun_model_name=mlrun_model_name,
            execution_mechanism="asyncio",
            image=self.image,
            requirements=["openai==1.77.0"],
            default_config={"max_tokens": 60},
            streaming=True,
        )
        function.deploy()

        url = function.get_url()
        resp = requests.post(
            f"{url}/v2/models/{mlrun_model_name}/infer",
            data=json.dumps(BATCH_INPUT_DATA[0]),
            stream=True,
            verify=mlrun.mlconf.httpdb.http.verify,
        )
        assert resp.ok, f"Streaming request failed: {resp.status_code} {resp.text}"
        assert resp.headers.get("Transfer-Encoding") == "chunked"

        chunks = list(resp.iter_content(decode_unicode=True, chunk_size=1024))
        assert len(chunks) > 1, "Expected multiple streamed chunks"
        full_text = "".join(chunks)
        assert EXPECTED_RESULTS[0] in full_text.lower()
        encoding = tiktoken.encoding_for_model(self.basic_llm_model)
        token_count = len(encoding.encode(full_text))
        assert 50 <= token_count <= 70
