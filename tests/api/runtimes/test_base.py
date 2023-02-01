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
#
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.errors
from mlrun.runtimes.base import BaseRuntime
from tests.api.runtimes.base import TestRuntimeBase


class TestBaseRunTime(TestRuntimeBase):
    def custom_setup_after_fixtures(self):
        self._mock_create_namespaced_pod()

    @pytest.mark.parametrize(
        "inputs", [{"input1": 123}, {"input1": None}, {"input1": None, "input2": 2}]
    )
    def test_run_with_invalid_inputs(self, db: Session, client: TestClient, inputs):
        runtime = BaseRuntime()
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentTypeError):
            self.execute_function(runtime, inputs=inputs)

    def test_run_with_valid_inputs(self, db: Session, client: TestClient):
        inputs = {"input1": "mlrun"}
        runtime = BaseRuntime()
        self.execute_function(runtime, inputs=inputs)
