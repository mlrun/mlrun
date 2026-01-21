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

from time import sleep

import pytest

import mlrun
from tests.datastore.remote_model.remote_model_utils import (
    setup_remote_model_test,
)
from tests.datastore.remote_model.test_remote_model import BaseMockModelProviderTest
from tests.system.model_monitoring import TestMLRunSystemModelMonitoring


class TestMockModelProviderTracking(
    BaseMockModelProviderTest, TestMLRunSystemModelMonitoring
):
    """Test MockModelProvider with tracking using real function deployment"""

    project_name = "mock-model-tracking-test"
    image = "artifactory.iguazeng.com:10557/tomerm/mlrun:llmodel_batch"

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_tracking(self, execution_mechanism):
        """Test single and batch invocations with MockModelProvider with model monitoring"""
        mlrun_model_name = "mock_model"
        endpoint_name = "my_endpoint"
        model_url = "mock://my-mock-model"

        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            model_url,
            mlrun_model_name=mlrun_model_name,
            image=self.image,
            execution_mechanism=execution_mechanism,
        )

        # Enable model monitoring
        self.set_mm_credentials()
        function.set_tracking()
        self.project.enable_model_monitoring(
            deploy_histogram_data_drift_app=False,
            image=self.image,
        )

        function.deploy()

        # Test 1: Single invocation
        self._check_single_invocation(function.invoke, mlrun_model_name)

        # Test 2: Batch invocation
        self._check_batch_invocation(function.invoke, mlrun_model_name)

        # Test 3: Single invocation with error
        self._check_single_invocation_with_error(function.invoke, mlrun_model_name)

        # Test 4: Batch invocation with error
        self._check_batch_invocation_with_error(function.invoke, mlrun_model_name)

        # Wait for monitoring data to be written
        sleep(5)

        # Verify model endpoint was created and tracked
        endpoint = (
            mlrun.get_run_db()
            .list_model_endpoints(
                self.project_name, metric_list=["error_count"], tsdb_metrics=True
            )
            .endpoints[0]
        )

        # Verify endpoint name
        assert endpoint.metadata.name == endpoint_name

        # Wait for metrics to be processed
        sleep(180)

        # Get model endpoint with feature analysis
        mep = mlrun.db.get_run_db().get_model_endpoint(
            name=endpoint_name,
            project=self.project.name,
            function_name=function.metadata.name,
            function_tag="latest",
            feature_analysis=True,
            tsdb_metrics=True,
        )

        # Verify monitoring data was captured for batch invocation
        assert mep is not None
