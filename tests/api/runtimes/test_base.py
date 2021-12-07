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
            self._execute_run(runtime, inputs=inputs)

    def test_run_with_valid_inputs(self, db: Session, client: TestClient):
        inputs = {"input1": "mlrun"}
        runtime = BaseRuntime()
        self._execute_run(runtime, inputs=inputs)
