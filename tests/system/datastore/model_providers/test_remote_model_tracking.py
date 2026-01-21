# Copyright 2026 Iguazio
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

import pytest

from mlrun.datastore.model_provider.model_provider import UsageResponseKeys
from tests.datastore.remote_model.remote_model_utils import (
    INPUT_DATA,
    setup_remote_model_test,
)
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestMockModelProviderTracking(TestMLRunSystem):
    """Test MockModelProvider with tracking using real function deployment"""

    project_name = "mock-model-tracking-test"
    image = "artifactory.iguazeng.com:10557/tomerm/mlrun:llmodel_batch"


    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_tracking(self, execution_mechanism):
        """Test single and batch invocations with MockModelProvider"""
        mlrun_model_name = "mock_model"
        model_url = "mock://my-mock-model"

        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            model_url,
            mlrun_model_name=mlrun_model_name,
            image=self.image,
            execution_mechanism=execution_mechanism,
        )
        function.deploy()

        # Test 1: Single invocation
        response = function.invoke(
            f"v2/models/{mlrun_model_name}/infer",
            json.dumps(INPUT_DATA[0]),
        )["output"]

        # Verify single response structure
        assert len(response) == 2  # answer + usage
        answer = response[UsageResponseKeys.ANSWER]
        stats = response[UsageResponseKeys.USAGE]

        # Verify mock message (no counter for single invocation)
        assert "mock model provider" in answer.lower()
        assert "(Item" not in answer  # No counter for single invocation

        # Verify mock usage stats (should be 0)
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0
        assert stats["total_tokens"] == 0

        # Test 2: Batch invocation
        batch_response = function.invoke(
            f"v2/models/{mlrun_model_name}/infer",
            json.dumps(INPUT_DATA),
        )

        # Assert we got list of 5 responses
        assert isinstance(batch_response, list)
        assert len(batch_response) == len(INPUT_DATA)

        # Verify each response has correct structure
        for i, full_result in enumerate(batch_response):
            result = full_result["output"]
            assert len(result) == 2  # answer + usage

            # Get answer and usage
            answer = result[UsageResponseKeys.ANSWER]
            stats = result[UsageResponseKeys.USAGE]

            # Verify mock message includes item index
            assert f"(Item {i})" in answer
            assert "mock model provider" in answer.lower()

            # Verify mock usage stats (should be 0)
            assert stats["prompt_tokens"] == 0
            assert stats["completion_tokens"] == 0
            assert stats["total_tokens"] == 0

