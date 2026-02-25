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
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pytest
import requests
import v3io
from storey import MapClass
from v3io.dataplane import RaiseForStatus

import mlrun
import mlrun.common.schemas
import mlrun.runtimes.mounts
import tests.system.base
from mlrun import feature_store as fstore
from mlrun.datastore.sources import KafkaSource
from mlrun.datastore.targets import ParquetTarget
from mlrun.runtimes.nuclio.function import AsyncSpec
from mlrun.serving import ModelRunnerStep
from mlrun.serving.remote import MLRunAPIRemoteStep, RemoteStep
from tests.system.model_monitoring import TestMLRunSystemModelMonitoring
from tests.system.runtimes.assets.function_llm_with_tools import MySelector
from tests.system.runtimes.assets.function_with_llm import MyLLM
from tests.system.runtimes.assets.function_with_model import DummyModel, MyModelSelector


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestNuclioRuntime(TestMLRunSystemModelMonitoring):
    project_name = "test-nuclio-runtime"

    image: str = "mlrun/mlrun"

    def test_deploy_function_with_error_handler(self):
        code_path = str(self.assets_path / "function-with-catcher.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function-with-catcher",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        graph = function.set_topology("flow", engine="async")
        graph.to(name="step1", handler="inc")
        graph.error_handler("catcher", handler="catcher", full_event=True)

        self._logger.debug("Deploying nuclio function")
        deployment = function.deploy()

        assert deployment == function.get_url()  # check function url

    def test_mlrun_project_accessibility(self):
        fn = mlrun.code_to_function(
            filename=str(self.assets_path / "nuclio_mlrun_function.py"),
            name="nuclio-mlrun",
            kind="nuclio",
            image=self.image,
            handler="my_func",
            project=self.project_name,
        )
        fn.deploy()
        response = fn.invoke(path="/")
        response_body = json.loads(response.decode("utf-8"))
        assert response_body.get("metadata", {}).get("name") == self.project_name

    @pytest.mark.parametrize("raise_exception", [True, False])
    @pytest.mark.parametrize("with_object", [True, False])
    def test_deploy_function_with_model_runner(self, raise_exception, with_object):
        code_path = str(self.assets_path / "function_with_model.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function_with_model",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(
            name="model-runner", raise_exception=raise_exception
        )
        if with_object:
            dummy_model = DummyModel(name="my-model")
        else:
            dummy_model = "DummyModel"
        model_runner_step.add_model(
            model_class=dummy_model,
            execution_mechanism="naive",
            endpoint_name="my-model",
        )

        graph.to(model_runner_step).respond()

        self._logger.debug("Deploying nuclio function")
        deployment = function.deploy()

        assert deployment == function.get_url()  # check function url

        resp = function.invoke("/", {"x": "y"})
        assert resp == {"x": "y", "extra": 123}

    @pytest.mark.parametrize("with_object", [True, False])
    def test_deploy_function_with_model_runner_with_selector(self, with_object):
        code_path = str(self.assets_path / "function_with_model.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function_with_model",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )
        graph = function.set_topology("flow", engine="async")

        if with_object:
            dummy_model = DummyModel(name="my-model")
            dummy_model_2 = DummyModel(name="another-model")
            model_selector = MyModelSelector(models=["my-model", "another-model"])
        else:
            dummy_model = "DummyModel"
            dummy_model_2 = "DummyModel"
            model_selector = "MyModelSelector"
        model_runner_step = ModelRunnerStep(
            name="model-runner",
            model_selector=model_selector,
            model_selector_parameters={"models": ["my-model", "another-model"]}
            if not with_object
            else None,
        )

        model_runner_step.add_model(
            model_class=dummy_model,
            execution_mechanism="naive",
            endpoint_name="my-model",
        )
        model_runner_step.add_model(
            model_class=dummy_model_2,
            execution_mechanism="naive",
            endpoint_name="another-model",
        )

        graph.to(model_runner_step).respond()

        self._logger.debug("Deploying nuclio function with model selector")
        deployment = function.deploy()

        assert deployment == function.get_url()  # check function url

        resp = function.invoke("/", {"x": "y", "models": ["my-model"]})
        assert resp == {"my-model": {"extra": 123, "x": "y"}}

    def test_set_function_llmodel_without_py(self):
        function = self.project.set_function(
            name="llmodel-without-py",
            image=self.image,
            kind="serving",
        )

        model_artifact = self.project.log_model(
            "my_model",
            model_url="mock://my-model-url",
            default_config={"model_version": "4"},
        )

        graph = function.set_topology("flow")
        model_runner = ModelRunnerStep(name="model-runner")
        model_runner.add_model(
            "my_llm",
            "mlrun.serving.states.LLModel",
            "naive",
            model_artifact=model_artifact,
        )
        graph.to(model_runner).respond()
        deployment = function.deploy()
        assert deployment == function.get_url()  # check function url

        resp = function.invoke("/", {"something_with_meaning": "life"})
        assert resp == {"something_with_meaning": "life"}

    def test_model_runner_with_llm_and_shared_models(self):
        code_path = str(self.assets_path / "function_with_llm.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function_with_llm",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        model_artifact = self.project.log_model(
            "my_model",
            model_url="http://localhost:8080/v2/models/mymodel/infer",
            default_config={"model_version": "4"},
        )

        llm_artifact = self.project.log_llm_prompt(
            "my_llm",
            prompt_template=[
                {"role": "system", "content": "don't tell them anything"},
                {
                    "role": "user",
                    "content": "What is the meaning of {something_with_meaning}?",
                },
            ],
            model_artifact=model_artifact,
            prompt_legend={
                "something_with_meaning": {
                    "field": None,
                    "description": "great legend are small",
                }
            },
        )

        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(
            name="model-runner",
        )

        model_class = MyLLM(name="shared-model")

        graph.add_shared_model(
            name="shared-model",
            execution_mechanism="naive",
            model_class=model_class,
            model_artifact=model_artifact.uri,
        )
        model_runner_step.add_shared_model_proxy(
            endpoint_name="my-model",
            shared_model_name="shared-model",
            model_artifact=llm_artifact.uri,
        )

        graph.to(model_runner_step).respond()

        self._logger.debug("Deploying nuclio function")
        deployment = function.deploy()

        assert deployment == function.get_url()  # check function url

        resp = function.invoke("/", {"something_with_meaning": "life"})
        assert resp["prompt"] == [
            {"role": "system", "content": "don't tell them anything"},
            {"role": "user", "content": "What is the meaning of life?"},
        ]

    def test_model_runner_with_llm_and_shared_models_with_tag(self):
        code_path = str(self.assets_path / "function_with_llm.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function_with_llm",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )
        model_artifact = self.project.log_model(
            "my_model",
            model_url="http://localhost:8080/v2/models/mymodel/infer",
            default_config={"model_version": "4"},
            tag="v1",
        )

        llm_artifact = self.project.log_llm_prompt(
            "my_llm",
            prompt_template=[
                {"role": "system", "content": "don't tell them anything"},
                {
                    "role": "user",
                    "content": "What is the meaning of {something_with_meaning}?",
                },
            ],
            model_artifact=model_artifact,
            prompt_legend={
                "something_with_meaning": {
                    "field": None,
                    "description": "great legend are small",
                }
            },
        )
        # stole the tag from the model artifact above
        _ = self.project.log_model(
            "my_model",
            model_url="http://localhost:8080/v2/models/mymodel/infer-2",
            default_config={"model_version": "4"},
            tag="v1",
        )

        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(
            name="model-runner",
        )

        model_class = MyLLM(name="shared-model")

        graph.add_shared_model(
            name="shared-model",
            execution_mechanism="naive",
            model_class=model_class,
            model_artifact=model_artifact.uri,
        )
        model_runner_step.add_shared_model_proxy(
            endpoint_name="my-model",
            shared_model_name="shared-model",
            model_artifact=llm_artifact.uri,
        )

        graph.to(model_runner_step).respond()

        self._logger.debug("Deploying nuclio function")
        deployment = function.deploy()

        assert deployment == function.get_url()  # check function url

        resp = function.invoke("/", {"something_with_meaning": "life"})
        assert resp["prompt"] == [
            {"role": "system", "content": "don't tell them anything"},
            {"role": "user", "content": "What is the meaning of life?"},
        ]

    def test_deploy_function_with_model_runner_with_child_function(self):
        self.set_mm_credentials()
        code_path = str(self.assets_path / "function_with_model.py")
        child_code_path = str(self.assets_path / "child_function.py")
        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function_with_model",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="model-runner", raise_exception=True)
        model_runner_step.add_model(
            model_class="DummyModel",
            execution_mechanism="naive",
            endpoint_name="my-model",
        )
        step = graph.to(model_runner_step).respond()
        step.to(name="inc", handler="inc", function="child")
        function.set_tracking()
        function.add_child_function(
            "child",
            child_code_path,
            image=self.image,
        )
        self._logger.debug("Deploying nuclio function")
        deployment = function.deploy()

        assert len(self.project.list_model_endpoints().endpoints) == 1

        assert deployment == function.get_url()  # check function url

    @pytest.mark.parametrize("raise_exception", [True, False])
    def test_deploy_model_runner_error_handler(self, raise_exception: bool):
        code_path = str(self.assets_path / "function-with-catcher.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function-with-catcher",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(
            name="model-runner", raise_exception=raise_exception
        )
        model_runner_step.add_model(
            model_class="ErrorModel",
            execution_mechanism="naive",
            endpoint_name="my-model",
        )

        step = graph.to(model_runner_step).respond()
        step.error_handler("catcher", handler="catcher_echo", full_event=True)

        self._logger.debug("Deploying nuclio function")
        deployment = function.deploy()

        assert deployment == function.get_url()  # check function url
        resp = function.invoke("/", {"x": "y"})
        assert (
            resp
            == {
                "error": "catcher_echo",
            }
            if raise_exception
            else {"error": "RuntimeError: "}
        )

    # Nuclio sometimes passes b'' instead of None due to dirty memory
    def test_workaround_for_nuclio_bug(self):
        code_path = str(self.assets_path / "nuclio_function_to_print_type.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="nuclio-bug-workaround-test-function",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        graph = function.set_topology("flow", engine="sync")
        graph.add_step(name="type", class_name="Type")

        self._logger.debug("Deploying nuclio function")
        url = function.deploy()

        for _ in range(10):
            resp = requests.get(url)
            assert resp.status_code == 200
            assert resp.text == "NoneType"

        for _ in range(10):
            resp = requests.post(url, data="abc")
            assert resp.status_code == 200
            assert resp.text == "bytes"

        for _ in range(10):
            resp = requests.get(url)
            assert resp.status_code == 200
            assert resp.text == "NoneType"

    def test_nuclio_function_status_fields(self):
        code_path = str(self.assets_path / "nuclio_function_to_print_type.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="test-function",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        # since we're deploying a serving function, we need to add a graph to it
        graph = function.set_topology("flow", engine="sync")
        graph.add_step(name="type", class_name="Type")

        self._logger.debug("Deploying nuclio function")
        url = function.deploy()

        resp = requests.get(url)
        assert resp.status_code == 200

        response = self._run_db.api_call(
            "GET",
            f"projects/{self.project_name}/functions",
        )

        assert response.ok
        data = response.json()
        deployed_function = data["funcs"][0]

        status = deployed_function["status"]
        assert "state" in status
        assert "nuclio_name" in status
        assert "internal_invocation_urls" in status
        assert "external_invocation_urls" in status
        assert "address" in status
        assert "container_image" in status

    # ML-3804
    def test_nuclio_function_handler_with_context(self):
        code_path = str(self.assets_path / "nuclio_function_with_context.py")

        serving_func_handler = self.project.set_function(
            name="serving-handler-func",
            func=code_path,
            image=self.image,
            kind="serving",
        )
        serving_func_handler.spec.parameters = {"Test": "test"}
        graph = serving_func_handler.set_topology("flow")
        graph.to(name="test", handler="test").respond()

        serving_func_deploy = self.project.deploy_function("serving-handler-func")

        serving_func_deploy.function.invoke("/")

    def test_nuclio_function_handler_with_batching(self):
        code_path = str(self.assets_path / "nuclio_function_batching.py")

        function = self.project.set_function(
            name="batching-handler-func",
            func=code_path,
            image=self.image,
            kind="nuclio",
        )
        function.with_http(
            batching_spec=mlrun.common.schemas.BatchingSpec(
                enabled=True, size=5, timeout="5s"
            )
        )
        function.deploy()

        with ThreadPoolExecutor(max_workers=10) as pool:
            responses = list(pool.map(lambda _: function.invoke("/"), range(10)))

        assert len(responses) == 10

        for resp in responses:
            assert b"Hello" in resp

        # unique IDs expected
        assert len(set(responses)) == 10

    @pytest.mark.parametrize("async_mode", [True, False])
    async def test_list_mep_through_api_step(self, async_mode: bool):
        code_path = str(self.assets_path / "nuclio_function.py")

        # Create serving function with MLRunAPIRemoteStep
        function = mlrun.code_to_function(
            name="api-list-meps-function",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        # Set up graph with MLRunAPIRemoteStep
        graph = function.set_topology("flow", engine="async" if async_mode else "sync")
        endpoint_path = f"projects/{self.project_name}/model-endpoints"
        graph.to(
            MLRunAPIRemoteStep(
                method=mlrun.common.types.HTTPMethod.GET,
                path=endpoint_path,
            ),
        ).respond()

        # # Deploy the function
        function.deploy()

        # Test event generation with sample event data
        event_data = {
            "params": {
                "function-name": "api-event-function",
                "function-tag": "latest",
                "tsdb-metrics": "False",
                "top-level": "False",
                "latest-only": "False",
            }
        }

        # Invoke function with event data
        resp = function.invoke("/", event_data)
        print(resp)

        # Verify event was generated by checking response
        assert resp is not None
        assert isinstance(resp, dict)
        assert "endpoints" in resp
        assert isinstance(resp["endpoints"], list)

    def test_serving_with_cyclic_graph(self):
        code_path = str(self.assets_path / "cyclic_function.py")
        function = mlrun.code_to_function(
            name="function-with-cyclic-graph",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )
        graph = function.set_topology(
            "flow", engine="async", allow_cyclic=True, max_iterations=6
        )
        graph.to(class_name="Counter", name="count").to(
            name="route", class_name="Route", cycle_to="count", end="Complete"
        ).respond()
        # Deploy the function
        function.deploy()

        resp = function.invoke(path="/", body={"counter": 1})
        assert resp["counter"] == 5
        with pytest.raises(
            RuntimeError, match=r"Max iterations exceeded in step 'count'"
        ):
            function.invoke(path="/", body={"counter": -5})

    @pytest.mark.parametrize(
        "execution_mechanism",
        ("naive", "thread_pool", "asyncio", "process_pool", "dedicated_process"),
    )
    def test_streaming_serving_function(self, execution_mechanism):
        """Test that streaming serving functions return chunked HTTP responses.

        Tests both StreamingStep (async generator do() method) and ModelRunnerStep
        with StreamingModel (generator predict() method) via a choice.
        """
        code_path = str(self.assets_path / "streaming_function.py")
        function = mlrun.code_to_function(
            name="streaming-function",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        # Build a graph with two branches via StreamingChoice:
        # - "step" route -> StreamingStep (tests async generator do())
        # - "model_runner" route -> ModelRunnerStep with StreamingModel (tests generator predict())
        # Both branches merge into a single responder
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="model_runner")
        model_runner_step.add_model(
            model_class="StreamingModel",
            execution_mechanism=execution_mechanism,
            endpoint_name="streaming_model",
            num_chunks=3,
        )
        choice = graph.to(name="choice", class_name="StreamingChoice")
        choice.to(name="step", class_name="StreamingStep", num_chunks=3)
        choice.to(model_runner_step)
        graph.add_step(
            name="responder",
            class_name="Echo",
            after=["step", "model_runner"],
        ).respond()

        function.set_streaming(enabled=True)
        function.deploy()

        url = function.get_url()

        # Test 1: StreamingStep path (async generator do() method)
        self._logger.info("Testing StreamingStep path...")
        resp = requests.post(f"{url}/step", data="test", stream=True)
        self._logger.info(f"StreamingStep response: {resp}")
        assert resp.ok, f"StreamingStep request failed: {resp.status_code} {resp.text}"
        assert resp.headers.get("Transfer-Encoding") == "chunked"

        chunks = list(resp.iter_content(decode_unicode=True, chunk_size=1024))
        self._logger.info(f"StreamingStep chunks: {chunks}")
        assert chunks == ["test_chunk_0", "test_chunk_1", "test_chunk_2"]

        # Test 2: ModelRunnerStep path (generator predict() method)
        self._logger.info("Testing ModelRunnerStep path...")
        resp = requests.post(f"{url}/model_runner", data="test", stream=True)
        self._logger.info(f"ModelRunnerStep response: {resp}")
        assert resp.ok, (
            f"ModelRunnerStep request failed: {resp.status_code} {resp.text}"
        )
        assert resp.headers.get("Transfer-Encoding") == "chunked"

        chunks = list(resp.iter_content(decode_unicode=True, chunk_size=1024))
        self._logger.info(f"ModelRunnerStep chunks: {chunks}")
        assert chunks == ["test_chunk_0", "test_chunk_1", "test_chunk_2"]

    def test_stream_response_termination_on_error(self):
        """
        Test that a mid-stream error terminates the stream, and that subsequent requests are served normally.

        Requires Nuclio 1.15.15 or later, which includes the fix for NUC-723 that caused hanging on mid-stream errors
        and failure to serve subsequent requests.

        Deploys a function with two routes:
        - /error  -> ErrorStreamingStep (yields one chunk then raises)
        - /healthy -> StreamingStep (normal streaming)

        Verifies the error request completes (does not hang) and a subsequent
        healthy request succeeds (worker still alive).
        """
        code_path = str(self.assets_path / "streaming_function.py")
        function = mlrun.code_to_function(
            name="streaming-error",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )
        function.spec.replicas = 1

        graph = function.set_topology("flow", engine="async")
        choice = graph.to(name="choice", class_name="StreamingChoice")
        choice.to(name="error", class_name="ErrorStreamingStep")
        choice.to(name="step", class_name="StreamingStep", num_chunks=3)
        graph.add_step(
            name="responder",
            class_name="Echo",
            after=["error", "step"],
        ).respond()

        function.set_streaming(enabled=True)
        function.deploy()

        url = function.get_url()

        # 1. Send a streaming request that triggers a mid-stream error.
        #    Use a timeout to guard against hangs (the pre-NUC-723 failure mode).
        self._logger.info("Sending error streaming request...")
        resp = requests.post(f"{url}/error", data="test", stream=True, timeout=30)

        try:
            chunks = list(resp.iter_content(decode_unicode=True, chunk_size=1024))
            self._logger.info(f"Error path chunks: {chunks}")
        except requests.exceptions.ChunkedEncodingError:
            self._logger.info(
                "Got ChunkedEncodingError (expected — stream terminated by server)"
            )

        # 2. Verify the worker is still healthy by sending a normal request.
        self._logger.info("Sending healthy streaming request...")
        resp = requests.post(f"{url}/step", data="test", stream=True, timeout=30)
        assert resp.ok, f"Healthy request failed: {resp.status_code} {resp.text}"
        assert resp.headers.get("Transfer-Encoding") == "chunked"

        chunks = list(resp.iter_content(decode_unicode=True, chunk_size=1024))
        self._logger.info(f"Healthy path chunks: {chunks}")
        assert chunks == ["test_chunk_0", "test_chunk_1", "test_chunk_2"]

    @pytest.mark.parametrize("with_object", [True, False])
    def test_mrs_with_tools_routing_sys(self, with_object):
        code_path = str(self.assets_path / "function_llm_with_tools.py")
        function = mlrun.code_to_function(
            name="llm-wih-tools",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )
        graph = function.set_topology("flow", engine="async", allow_cyclic=True)
        if with_object:
            model_runner_step = ModelRunnerStep(
                name="my_model_runner",
                model_runner_selector=MySelector(tool_a="tool_a", tool_b="tool_b"),
            )
        else:
            model_runner_step = ModelRunnerStep(
                name="my_model_runner",
                model_runner_selector="MySelector",
                model_runner_selector_parameters={
                    "tool_a": "tool_a",
                    "tool_b": "tool_b",
                },
            )
        model_runner_step.add_model(
            model_class="LLModelWithTools",
            execution_mechanism="naive",
            endpoint_name="llm_with_tools",
        )
        runner = graph.to(name="start", class_name="Echo").to(model_runner_step)
        runner.to(name="tool_a", class_name="Tool", cycle_to="my_model_runner")
        runner.to(name="tool_b", class_name="Tool", cycle_to="my_model_runner")
        runner.to(name="end", class_name="Echo").respond()

        # Deploy the function
        function.deploy()

        resp = function.invoke(path="/", body={"counter": 0})
        assert resp["counter"] == 5
        assert resp["tool_a"] == 2
        assert resp["tool_b"] == 2

    def test_async_http_mode(self):
        code_path = str(self.assets_path / "async_nuclio_func.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="async-http-function",
            kind="nuclio",
            project=self.project_name,
            filename=code_path,
            image=self.image,
            handler="main:async_handler",
        )
        function.spec.function_handler = "main:async_handler"

        function.with_http(async_spec=AsyncSpec(enabled=True, max_connections=100))

        self._logger.debug("Deploying nuclio function")
        function.deploy()

        self._logger.debug("Triggering nuclio function")
        start = time.time()
        with ThreadPoolExecutor(max_workers=100) as executor:
            # Submit tasks
            futures = [
                executor.submit(function.invoke, path="/", body=[i]) for i in range(100)
            ]
            # Retrieve results as they complete
            for future in as_completed(futures):
                future.result()
        end = time.time()
        timing = end - start
        assert timing < 7, (
            f"running nuclio async mode took {timing} seconds should be < 7"
        )

    @pytest.mark.parametrize("with_code", [True, False])
    def test_async_http_mode_serving_graph(self, with_code):
        async_code_path = str(self.assets_path / "async_serving_func.py")
        code_path = str(self.assets_path / "async_nuclio_func.py")

        self._logger.debug("Creating serving function")
        project = mlrun.get_or_create_project(
            self.project_name, allow_cross_project=True
        )
        nuclio_function = project.set_function(
            func=code_path,
            name="serving-function",
            kind="nuclio",
            image=self.image,
            handler="main:async_handler",
        )
        nuclio_function.spec.function_handler = "main:async_handler"
        nuclio_function.with_http(
            async_spec=AsyncSpec(enabled=True, max_connections=200)
        )
        url = nuclio_function.deploy()
        async_function = project.set_function(
            func=async_code_path if with_code else None,
            name="remote-http",
            kind="serving",
            image=self.image,
        )

        graph = async_function.set_topology("flow", engine="async")
        graph.to(
            RemoteStep(
                name="remote_echo",
                url=url,
                body_expression="event['inputs']",
                result_path="resp",
                retries=0,
                max_in_flight=16,
                timeout=100,
            )
        ).respond()

        async_function.with_http(
            async_spec=AsyncSpec(enabled=True, max_connections=200)
        )

        self._logger.debug("Deploying nuclio function")
        async_function.deploy()

        self._logger.debug("Triggering async serving function")
        start = time.time()
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Submit tasks
            futures = [
                executor.submit(
                    async_function.invoke, path="/", body={"inputs": [[1, 2], [1, 2]]}
                )
                for i in range(16)
            ]
            # Retrieve results as they complete
            for future in as_completed(futures):
                future.result()
        end = time.time()
        timing = end - start
        assert timing < 7, (
            f"running serving async mode took {timing} seconds should be < 7"
        )


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestNuclioRuntimeWithStream(tests.system.base.TestMLRunSystem):
    project_name = "stream-project"
    stream_container = "bigdata"
    path_uuid_part = uuid.uuid4()
    stream_path = f"/test_nuclio/test_serving_with_child_function-{path_uuid_part}/"
    stream_path_out = (
        f"/test_nuclio/test_serving_with_child_function_out-{path_uuid_part}/"
    )

    image: str = "mlrun/mlrun"

    def custom_teardown(self):
        v3io_client = v3io.dataplane.Client(
            endpoint=os.environ["V3IO_API"], access_key=os.environ["V3IO_ACCESS_KEY"]
        )
        v3io_client.stream.delete(
            self.stream_container,
            self.stream_path,
            raise_for_status=RaiseForStatus.never,
        )

    def test_serving_with_child_function(self):
        code_path = str(self.assets_path / "nuclio_function.py")
        child_code_path = str(self.assets_path / "child_function.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function-with-child",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        graph = function.set_topology("flow", engine="async")

        graph.to(
            ">>",
            "q1",
            path=f"v3io:///{self.stream_container}{self.stream_path}",
            sharding_func=1,
            shards=3,
            full_event=True,
        ).to(name="child", class_name="Identity", function="child").to(
            ">>",
            "out",
            path=f"/{self.stream_container}{self.stream_path_out}",
            sharding_func=2,
            shards=3,
        )

        graph.add_step(
            name="otherchild",
            class_name="Augment",
            after="q1",
            function="otherchild",
            full_event=True,
        )

        graph["out"].after_step("otherchild")

        function.add_child_function(
            "child",
            child_code_path,
            image=self.image,
        )
        function.add_child_function(
            "otherchild",
            child_code_path,
            image=self.image,
        )

        self._logger.debug("Deploying nuclio function")
        url = function.deploy()

        db_function = self.project.get_function(function.metadata.name)
        nuclio_config = db_function.spec.config

        for key, value in nuclio_config.items():
            if key.startswith("spec.triggers"):
                assert value.get("password", "").startswith(
                    mlrun.model.Credentials.secret_reference_prefi
                )

        self._logger.debug("Triggering nuclio function")
        resp = requests.post(url, json={"hello": "world"})
        assert resp.status_code == 200

        time.sleep(10)

        client = v3io.dataplane.Client()

        resp = client.stream.seek(
            self.stream_container, self.stream_path_out, 2, "EARLIEST"
        )
        self._logger.debug(f"Out stream Seek response: {resp.status_code}: {resp.body}")
        location = json.loads(resp.body.decode("utf8"))["Location"]
        resp = client.stream.get_records(
            self.stream_container, self.stream_path_out, shard_id=2, location=location
        )
        self._logger.debug(
            f"Out stream GetRecords response: {resp.status_code}: {resp.body}"
        )
        assert resp.status_code == 200

        assert len(resp.output.records) == 2
        record1, record2 = resp.output.records

        expected_record = b'{"hello": "world"}'
        expected_other_record = b'{"hello": "world", "more_stuff": 5, "path": "/"}'

        assert (
            record1.data == expected_record and record2.data == expected_other_record
        ) or (record2.data == expected_record and record1.data == expected_other_record)

        resp = client.stream.seek(
            self.stream_container, self.stream_path, 1, "EARLIEST"
        )
        self._logger.debug(
            f"Intermediate stream Seek response: {resp.status_code}: {resp.body}"
        )
        location = json.loads(resp.body.decode("utf8"))["Location"]
        resp = client.stream.get_records(
            self.stream_container, self.stream_path, shard_id=1, location=location
        )
        self._logger.debug(
            f"Intermediate stream GetRecords response: {resp.status_code}: {resp.body}"
        )
        assert resp.status_code == 200
        assert len(resp.output.records) == 1
        record = resp.output.records[0]
        record = json.loads(record.data.decode("utf8"))
        self._logger.debug(f"Intermediate record: {record}")
        assert record["full_event_wrapper"] is True
        assert record["body"] == {"hello": "world"}
        assert "id" in record.keys()


class MyMap(MapClass):
    def do(self, event):
        self.context.logger.info(f"MyMap: event = {event}")

        if isinstance(event, bytes):
            event = {"key": event}
        return event


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestNuclioRuntimeWithKafka(tests.system.base.TestMLRunSystem):
    project_name = "kafka-project"
    topic_uuid_part = uuid.uuid4()
    topic = f"TestNuclioRuntimeWithKafka-{topic_uuid_part}"
    topic_out = f"TestNuclioRuntimeWithKafka-out-{topic_uuid_part}"
    brokers = os.getenv("MLRUN_SYSTEM_TESTS_KAFKA_BROKERS")

    image: str = "mlrun/mlrun"

    @pytest.fixture()
    def kafka_fixture(self):
        import kafka

        # Setup
        kafka_admin_client = kafka.KafkaAdminClient(bootstrap_servers=self.brokers)
        kafka_admin_client.create_topics(
            [
                kafka.admin.NewTopic(
                    self.topic, num_partitions=3, replication_factor=1
                ),
                kafka.admin.NewTopic(
                    self.topic_out, num_partitions=3, replication_factor=1
                ),
            ]
        )

        kafka_consumer = kafka.KafkaConsumer(
            self.topic_out,
            bootstrap_servers=self.brokers,
            auto_offset_reset="earliest",
            consumer_timeout_ms=10 * 60 * 1000,
        )

        kafka_producer = kafka.KafkaProducer(bootstrap_servers=self.brokers)

        # Test runs
        yield kafka_consumer, kafka_producer, kafka_admin_client

        # Teardown
        kafka_admin_client.delete_topics([self.topic, self.topic_out])
        kafka_admin_client.close()
        kafka_consumer.close()

    def produce_kafka_helper(self, kafka_producer, df):
        import io

        import avro.schema
        from avro.io import DatumWriter

        from .assets.map_avro import MyMap

        for row_index, _ in df.iterrows():
            event_row_temp = df.loc[[row_index]]
            event_row_dict = event_row_temp.to_dict("records")[0]
            writer = DatumWriter(MyMap.AVRO_SCHEMA)
            bytes_writer = io.BytesIO()
            encoder = avro.io.BinaryEncoder(bytes_writer)
            writer.write(
                event_row_dict,
                encoder,
            )
            raw_bytes = bytes_writer.getvalue()
            kafka_producer.send(self.topic_out, raw_bytes)
            kafka_producer.flush()

    @pytest.mark.skipif(
        not brokers, reason="MLRUN_SYSTEM_TESTS_KAFKA_BROKERS not defined"
    )
    def test_kafka_source_with_avro(self, kafka_fixture):
        row_divide = 3
        stocks_df = pd.DataFrame(
            {
                "ticker": ["MSFT", "GOOG", "AAPL", "CSCO", "META", "AMZN"],
                "name": [
                    "Microsoft Corporation",
                    "Alphabet Inc",
                    "Apple Inc",
                    "Cisco Systems Inc",
                    "Meta Platforms Inc",
                    "Amazon.com Inc",
                ],
                "price": [30, 40, 50, 20, 60, 70],
            }
        )
        fs_name = "stocks_set"
        stocks_set = fstore.FeatureSet(fs_name, entities=[fstore.Entity("ticker")])

        # need to set full_event=True since we need to change event key in the Map step
        stocks_set.graph.to("MyMap", full_event=True)

        target = ParquetTarget(flush_after_seconds=10)
        stocks_set.ingest(
            source=stocks_df[0:row_divide],
            targets=[target],
            infer_options=fstore.InferOptions.default(),
        )
        stocks_set.save()

        consumer_group = "my_group"

        kafka_source = KafkaSource(
            brokers=self.brokers,
            topics=self.topic_out,
            initial_offset="earliest",
            group=consumer_group,
        )

        func = mlrun.code_to_function(
            name="map",
            kind="serving",
            image=self.image,
            requirements=["avro"],
            filename=str(self.assets_path / "map_avro.py"),
        )

        func.spec.min_replicas = 1
        func.spec.max_replicas = 1

        run_config = fstore.RunConfig(local=False, function=func).apply(
            mlrun.runtimes.mounts.auto_mount()
        )
        stocks_set_endpoint, _ = stocks_set.deploy_ingestion_service(
            source=kafka_source,
            targets=[target],
            run_config=run_config,
        )
        print(stocks_set_endpoint)

        kafka_consumer, kafka_producer, kafka_admin = kafka_fixture
        self.produce_kafka_helper(kafka_producer, stocks_df[row_divide:])

        time.sleep(20)  # wait for ingestion-service parquet to be written

        vec = fstore.FeatureVector("test-vec", [f"{fs_name}.*"])
        resp = vec.get_offline_features(with_indexes=True)
        actual_df = resp.to_dataframe()

        print(actual_df)
        expected_df = stocks_df.set_index("ticker")
        # setting check_like=True since the order of the two parquet merge
        # can happen two-ways (based on alphanumeric order)
        pd.testing.assert_frame_equal(actual_df, expected_df, check_like=True)

        consumer_group_offsets = kafka_admin.list_consumer_group_offsets(consumer_group)
        print(f"consumer_group_offsets={consumer_group_offsets}")
        sum_of_offsets = 0
        for topic_partition, offset_and_metadata in consumer_group_offsets.items():
            if topic_partition.topic == self.topic_out:
                sum_of_offsets += offset_and_metadata.offset
        assert sum_of_offsets == 3

    @pytest.mark.skipif(
        not brokers, reason="MLRUN_SYSTEM_TESTS_KAFKA_BROKERS not defined"
    )
    def test_serving_with_kafka_queue(self, kafka_fixture):
        kafka_consumer, _, _ = kafka_fixture
        code_path = str(self.assets_path / "nuclio_function.py")
        child_code_path = str(self.assets_path / "child_function.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function-with-child-kafka",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        graph = function.set_topology("flow", engine="async")

        graph.to(
            ">>",
            "q1",
            path=f"kafka://{self.brokers}/{self.topic}",
            sharding_func=1,
            full_event=True,
        ).to(name="child", class_name="Identity", function="child").to(
            ">>",
            "out",
            path=self.topic_out,
            kafka_brokers=self.brokers,
            sharding_func=2,
        )

        graph.add_step(
            name="other-child",
            class_name="Augment",
            after="q1",
            function="other-child",
            full_event=True,
        )

        graph["out"].after_step("other-child")

        function.add_child_function(
            "child",
            child_code_path,
            image=self.image,
        )
        function.add_child_function(
            "other-child",
            child_code_path,
            image=self.image,
        )

        self._logger.debug("Deploying nuclio function")
        url = function.deploy()

        self._logger.debug("Triggering nuclio function")
        resp = requests.post(url, json={"hello": "world"})
        assert resp.status_code == 200

        expected_record = b'{"hello": "world"}'
        expected_other_record = b'{"hello": "world", "more_stuff": 5, "path": "/"}'

        self._logger.debug("Waiting for data to arrive in output topic")
        kafka_consumer.subscribe([self.topic_out])
        record1 = next(kafka_consumer)
        record2 = next(kafka_consumer)
        assert (
            record1.value == expected_record and record2.value == expected_other_record
        ) or (
            record2.value == expected_record or record1.value == expected_other_record
        )
        assert record1.partition == 2
        assert record2.partition == 2
        kafka_consumer.unsubscribe()

        # Intermediate record should have been written as a full event
        kafka_consumer.subscribe([self.topic])
        record = next(kafka_consumer)
        payload = json.loads(record.value.decode("utf8"))
        self._logger.debug(f"Intermediate record: {payload}")
        assert payload["full_event_wrapper"] is True
        assert payload["body"] == {"hello": "world"}
        assert "id" in payload.keys()
        assert record.partition == 1


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestNuclioMLRunJobs(tests.system.base.TestMLRunSystem):
    project_name = "nuclio-mlrun-jobs"

    image: str = "mlrun/mlrun"

    def _deploy_function(self, replicas=1):
        filename = str(self.assets_path / "handler.py")
        fn = mlrun.code_to_function(
            filename=filename,
            name="nuclio-mlrun",
            kind="nuclio:mlrun",
            image=self.image,
            handler="my_func",
        )
        # replicas * workers need to match or exceed parallel_runs
        fn.spec.replicas = replicas
        fn.with_http(workers=2)
        fn.deploy()
        return fn

    def test_single_run(self):
        fn = self._deploy_function()
        run_result = fn.run(params={"p1": 8})

        print(run_result.to_yaml())
        assert run_result.state() == "completed", "wrong state"
        # accuracy = p1 * 2
        assert run_result.output("accuracy") == 16, "unexpected results"

    def test_hyper_run(self):
        fn = self._deploy_function(2)

        hyper_param_options = mlrun.model.HyperParamOptions(
            parallel_runs=4,
            selector="max.accuracy",
            max_errors=1,
        )

        p1 = [4, 2, 5, 8, 9, 6, 1, 11, 1, 1, 2, 1, 1]
        run_result = fn.run(
            params={"p2": "xx"},
            hyperparams={"p1": p1},
            hyper_param_options=hyper_param_options,
        )
        print(run_result.to_yaml())
        assert run_result.state() == "completed", "wrong state"
        # accuracy = max(p1) * 2
        assert run_result.output("accuracy") == 22, "unexpected results"

        # Cover listing artifacts with partitioning when logging an artifact inside a run with hyperparameters
        artifacts = mlrun.get_run_db().list_artifacts(
            partition_by=mlrun.common.schemas.ArtifactPartitionByField.project_and_name,
            tag="latest",
        )
        assert len(artifacts) == 3  # iteration_results + parallel_coordinates + test
        for artifact in artifacts:
            # We are not checking the best iteration here because it is not guaranteed
            assert artifact["metadata"]["tag"] == "latest"

        # test early stop
        hyper_param_options = mlrun.model.HyperParamOptions(
            parallel_runs=1,
            selector="max.accuracy",
            max_errors=1,
            stop_condition="accuracy>9",
        )

        run_result = fn.run(
            params={"p2": "xx"},
            hyperparams={"p1": p1},
            hyper_param_options=hyper_param_options,
        )
        print(run_result.to_yaml())
        assert run_result.state() == "completed", "wrong state"
        # accuracy = max(p1) * 2, stop where accuracy > 9
        assert run_result.output("accuracy") == 10, "unexpected results"


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestNuclioAPIGateways(tests.system.base.TestMLRunSystem):
    project_name = "nuclio-mlrun-gateways"
    gw_name = "test-gateway"

    def custom_setup(self):
        self.f1 = self._deploy_function(suffix="1")
        self.f2 = self._deploy_function(suffix="2")

    def test_basic_api_gateway_flow(self):
        api_gateway = self._get_basic_gateway()
        api_gateway = self.project.store_api_gateway(api_gateway=api_gateway)
        res = api_gateway.invoke(verify=False)
        assert res.status_code == 200
        # check that api gateway url is in function's external_invocation_urls
        self._check_functions_external_invocation_urls(
            function_name=self.f1.metadata.name,
            expected_url=api_gateway.invoke_url.replace("https://", ""),
        )
        self._cleanup_gateway()

        api_gateway = self._get_basic_gateway()
        api_gateway.with_basic_auth("test", "test")
        api_gateway = self.project.store_api_gateway(api_gateway=api_gateway)
        res = api_gateway.invoke(credentials=("test", "test"), verify=False)
        assert res.status_code == 200

        # check that api gateway url is in function's external_invocation_urls
        self._check_functions_external_invocation_urls(
            function_name=self.f1.metadata.name,
            expected_url=api_gateway.invoke_url.replace("https://", ""),
        )
        self._cleanup_gateway()

        api_gateway = self._get_basic_gateway()
        api_gateway.with_canary(functions=[self.f1, self.f2], canary=[50, 50])
        api_gateway = self.project.store_api_gateway(api_gateway=api_gateway)
        res = api_gateway.invoke(verify=False)
        assert res.status_code == 200

        # check that api gateway url is in function's external_invocation_urls
        self._check_functions_external_invocation_urls(
            function_name=self.f1.metadata.name,
            expected_url=api_gateway.invoke_url.replace("https://", ""),
        )
        self._check_functions_external_invocation_urls(
            function_name=self.f2.metadata.name,
            expected_url=api_gateway.invoke_url.replace("https://", ""),
        )

    def _get_basic_gateway(self):
        return mlrun.runtimes.nuclio.api_gateway.APIGateway(
            metadata=mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(
                name=self.gw_name,
            ),
            spec=mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
                functions=[self.f1], project=self.project_name
            ),
        )

    def _cleanup_gateway(self):
        self.project.delete_api_gateway(self.gw_name)

    def _check_functions_external_invocation_urls(
        self, function_name: str, expected_url: str
    ):
        function = self.project.get_function(function_name)
        urls = function.to_dict().get("status", {}).get("external_invocation_urls")
        assert expected_url in urls

    def _deploy_function(self, replicas=1, suffix=""):
        filename = str(self.assets_path / "nuclio_function.py")

        fn = mlrun.code_to_function(
            filename=filename,
            name=f"nuclio-mlrun-{suffix}",
            kind="nuclio",
            image="python:3.9",
            handler="handler",
        )
        fn.spec.replicas = replicas
        fn.with_http(workers=1)
        fn.deploy()
        return fn


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestNuclioRuntimeWithRabbitMQ(tests.system.base.TestMLRunSystem):
    """System tests for RabbitMQ trigger support in Nuclio functions.

    Requires:
        - MLRUN_SYSTEM_TESTS_RABBITMQ_URL: RabbitMQ AMQP URL (e.g., amqp://user:pass@host:5672)
    """

    project_name = "rabbitmq-project"
    exchange_uuid_part = uuid.uuid4()
    exchange_name = f"test-exchange-{exchange_uuid_part}"
    queue_name = f"test-queue-{exchange_uuid_part}"
    rabbitmq_url = os.getenv("MLRUN_SYSTEM_TESTS_RABBITMQ_URL")

    image: str = "mlrun/mlrun"

    def _create_rabbitmq_connection(self):
        """Create a fresh RabbitMQ connection."""
        import pika

        params = pika.URLParameters(self.rabbitmq_url)
        # Increase heartbeat to handle longer operations
        params.heartbeat = 600
        connection = pika.BlockingConnection(params)
        return connection, connection.channel()

    @pytest.fixture()
    def rabbitmq_fixture(self):
        import tempfile

        # Create temp directory for parquet output
        tmp_dir = tempfile.mkdtemp(prefix="rabbitmq_test_")

        # Setup: Create exchange and queue, then close connection
        # This avoids heartbeat timeouts during long deployments
        connection, channel = self._create_rabbitmq_connection()
        channel.exchange_declare(
            exchange=self.exchange_name, exchange_type="topic", durable=False
        )
        channel.queue_declare(queue=self.queue_name, durable=False)
        channel.queue_bind(
            exchange=self.exchange_name, queue=self.queue_name, routing_key="test.#"
        )
        connection.close()

        try:
            # Yield a helper to create fresh connections for publishing
            yield self._create_rabbitmq_connection, tmp_dir
        finally:
            # Cleanup: delete queue, exchange, and temp directory
            try:
                connection, channel = self._create_rabbitmq_connection()
                channel.queue_delete(queue=self.queue_name)
                channel.exchange_delete(exchange=self.exchange_name)
                connection.close()
            except Exception:
                pass
            try:
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    @pytest.mark.skipif(
        not rabbitmq_url, reason="MLRUN_SYSTEM_TESTS_RABBITMQ_URL not defined"
    )
    def test_rabbitmq_trigger_end_to_end(self, rabbitmq_fixture):
        """Test end-to-end message processing with RabbitMQ trigger.

        Verifies:
        1. Function deploys successfully with RabbitMQ trigger
        2. Messages published to RabbitMQ are received and processed
        3. Processed messages are written to v3io Parquet output
        """
        create_connection, _ = rabbitmq_fixture

        # Use v3io path accessible from Kubernetes
        v3io_username = os.getenv("V3IO_USERNAME", "iguazio")
        output_path = (
            f"v3io:///users/{v3io_username}/rabbitmq_test_{self.exchange_uuid_part}"
        )

        code_path = str(self.assets_path / "rabbitmq_handler.py")

        self._logger.debug("Creating serving function with RabbitMQ trigger")
        function = mlrun.code_to_function(
            name="rabbitmq-e2e-test",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        # Set up graph: ProcessMessage -> ParquetTarget (v3io)
        graph = function.set_topology("flow", engine="async")
        graph.to(name="process", class_name="ProcessMessage").to(
            name="parquet",
            class_name="storey.ParquetTarget",
            path=output_path,
            flush_after_seconds=5,
        )

        # Add RabbitMQ trigger
        function.add_rabbitmq_trigger(
            url=self.rabbitmq_url,
            exchange_name=self.exchange_name,
            queue_name=self.queue_name,
            prefetch_count=1,
        )

        self._logger.debug("Deploying serving function")
        function.deploy()

        # Verify function is deployed and running
        db_function = self.project.get_function(function.metadata.name)
        assert db_function.status.state == "ready"

        # Create fresh connection after deployment to publish messages
        connection, channel = create_connection()
        try:
            # Publish test messages
            test_messages = [
                {"message_id": "msg1", "test": "hello_rabbitmq", "value": 42},
                {"message_id": "msg2", "test": "second_message", "value": 100},
            ]

            for msg in test_messages:
                channel.basic_publish(
                    exchange=self.exchange_name,
                    routing_key="test.message",
                    body=json.dumps(msg),
                )
                self._logger.debug(f"Published message: {msg['message_id']}")
        finally:
            connection.close()

        # Wait for processing and parquet flush
        self._logger.debug("Waiting for message processing and parquet write...")
        time.sleep(15)

        try:
            # Read parquet output from v3io and verify
            df = pd.read_parquet(output_path)
            self._logger.debug(f"Read {len(df)} records from parquet")

            assert len(df) >= 2, f"Expected at least 2 records, got {len(df)}"
            assert "_processed" in df.columns, "Missing _processed marker column"
            assert df["_processed"].all(), "Not all messages have _processed=True"

            # Verify message content
            message_ids = set(df["message_id"].tolist())
            assert "msg1" in message_ids, "msg1 not found in output"
            assert "msg2" in message_ids, "msg2 not found in output"
        finally:
            # Cleanup v3io output
            try:
                import fsspec

                fs = fsspec.filesystem("v3io")
                fs.rm(output_path.replace("v3io://", ""), recursive=True)
            except Exception as e:
                self._logger.warning(f"Failed to cleanup v3io output: {e}")

    @pytest.mark.skipif(
        not rabbitmq_url, reason="MLRUN_SYSTEM_TESTS_RABBITMQ_URL not defined"
    )
    def test_rabbitmq_trigger_with_topics(self, rabbitmq_fixture):
        """Test deploying a function with RabbitMQ trigger using topics."""
        create_connection, _ = rabbitmq_fixture

        code_path = str(self.assets_path / "rabbitmq_handler.py")

        self._logger.debug("Creating serving function with RabbitMQ trigger (topics)")
        function = mlrun.code_to_function(
            name="rabbitmq-binding-test",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        # Set up simple graph
        graph = function.set_topology("flow", engine="async")
        graph.to(name="process", class_name="ProcessMessage").respond()

        # Add RabbitMQ trigger with topics (creates a unique queue)
        function.add_rabbitmq_trigger(
            url=self.rabbitmq_url,
            exchange_name=self.exchange_name,
            topics=["events.#", "notifications.*"],
            prefetch_count=1,
        )

        self._logger.debug("Deploying serving function")
        function.deploy()

        # Verify function is deployed and running
        db_function = self.project.get_function(function.metadata.name)
        assert db_function.status.state == "ready"

        # Create fresh connection after deployment to publish messages
        connection, channel = create_connection()
        try:
            # Publish test messages to different routing keys
            for routing_key in ["events.user.created", "notifications.alert"]:
                test_message = json.dumps({"routing_key": routing_key, "data": "test"})
                channel.basic_publish(
                    exchange=self.exchange_name,
                    routing_key=routing_key,
                    body=test_message,
                )
                self._logger.debug(f"Published message with routing key: {routing_key}")
        finally:
            connection.close()

        # Allow time for message processing
        time.sleep(5)

    @pytest.mark.skipif(
        not rabbitmq_url, reason="MLRUN_SYSTEM_TESTS_RABBITMQ_URL not defined"
    )
    def test_rabbitmq_trigger_config_in_nuclio_spec(self, rabbitmq_fixture):
        """Test that RabbitMQ trigger configuration is correctly passed to Nuclio."""
        code_path = str(self.assets_path / "rabbitmq_handler.py")

        function = mlrun.code_to_function(
            name="rabbitmq-config-test",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image=self.image,
        )

        # Set up simple graph (required for serving function)
        graph = function.set_topology("flow", engine="async")
        graph.to(name="process", class_name="ProcessMessage").respond()

        # Add RabbitMQ trigger with various configurations
        function.add_rabbitmq_trigger(
            url=self.rabbitmq_url,
            exchange_name=self.exchange_name,
            queue_name=self.queue_name,
            prefetch_count=10,
            durable_exchange=True,
            durable_queue=True,
            on_error="nack",
            requeue_on_error=True,
            num_workers=2,
        )

        # Verify trigger configuration before deployment
        trigger_config = function.spec.config.get("spec.triggers.rabbitmq")
        assert trigger_config is not None
        assert trigger_config["kind"] == "rabbit-mq"
        assert trigger_config["attributes"]["exchangeName"] == self.exchange_name
        assert trigger_config["attributes"]["queueName"] == self.queue_name
        assert trigger_config["attributes"]["prefetchCount"] == 10
        assert trigger_config["attributes"]["durableExchange"] is True
        assert trigger_config["attributes"]["durableQueue"] is True
        assert trigger_config["attributes"]["onError"] == "nack"
        assert trigger_config["attributes"]["requeueOnError"] is True
        assert trigger_config["numWorkers"] == 2
