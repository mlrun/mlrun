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
import json
import os
import time
import uuid

import pytest
import requests
import v3io
from v3io.dataplane import RaiseForStatus

import mlrun
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestNuclioRuntime(tests.system.base.TestMLRunSystem):
    project_name = "does-not-exist-3"

    def test_deploy_function_with_error_handler(self):
        code_path = str(self.assets_path / "function-with-catcher.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function-with-catcher",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )

        graph = function.set_topology("flow", engine="async")

        graph.to(name="step1", handler="inc")
        graph.add_step(name="catcher", handler="catcher", full_event=True, after="")

        graph.error_handler("catcher")

        self._logger.debug("Deploying nuclio function")
        function.deploy()

    # Nuclio sometimes passes b'' instead of None due to dirty memory
    def test_workaround_for_nuclio_bug(self):
        code_path = str(self.assets_path / "nuclio_function_to_print_type.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="nuclio-bug-workaround-test-function",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
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

    def custom_teardown(self):
        v3io_client = v3io.dataplane.Client(
            endpoint=os.environ["V3IO_API"], access_key=os.environ["V3IO_ACCESS_KEY"]
        )
        v3io_client.delete_stream(
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
            image="mlrun/mlrun",
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
            name="otherchild", class_name="Augment", after="q1", function="otherchild"
        )

        graph["out"].after_step("otherchild")

        function.add_child_function(
            "child",
            child_code_path,
            image="mlrun/mlrun",
        )
        function.add_child_function(
            "otherchild",
            child_code_path,
            image="mlrun/mlrun",
        )

        self._logger.debug("Deploying nuclio function")
        url = function.deploy()

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
        expected_other_record = b'{"hello": "world", "more_stuff": 5}'

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
        assert "time" in record.keys()
        assert "id" in record.keys()


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestNuclioRuntimeWithKafka(tests.system.base.TestMLRunSystem):
    project_name = "kafka-project"
    topic_uuid_part = uuid.uuid4()
    topic = f"TestNuclioRuntimeWithKafka-{topic_uuid_part}"
    topic_out = f"TestNuclioRuntimeWithKafka-out-{topic_uuid_part}"
    brokers = os.getenv("MLRUN_SYSTEM_TESTS_KAFKA_BROKERS")

    @pytest.fixture()
    def kafka_consumer(self):
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

        # Test runs
        yield kafka_consumer

        # Teardown
        kafka_admin_client.delete_topics([self.topic, self.topic_out])
        kafka_admin_client.close()
        kafka_consumer.close()

    @pytest.mark.skipif(
        not brokers, reason="MLRUN_SYSTEM_TESTS_KAFKA_BROKERS not defined"
    )
    def test_serving_with_kafka_queue(self, kafka_consumer):
        code_path = str(self.assets_path / "nuclio_function.py")
        child_code_path = str(self.assets_path / "child_function.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function-with-child-kafka",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
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
            kafka_bootstrap_servers=self.brokers,
            sharding_func=2,
        )

        graph.add_step(
            name="other-child", class_name="Augment", after="q1", function="other-child"
        )

        graph["out"].after_step("other-child")

        function.add_child_function(
            "child",
            child_code_path,
            image="mlrun/mlrun",
        )
        function.add_child_function(
            "other-child",
            child_code_path,
            image="mlrun/mlrun",
        )

        self._logger.debug("Deploying nuclio function")
        url = function.deploy()

        self._logger.debug("Triggering nuclio function")
        resp = requests.post(url, json={"hello": "world"})
        assert resp.status_code == 200

        expected_record = b'{"hello": "world"}'
        expected_other_record = b'{"hello": "world", "more_stuff": 5}'

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
        assert "time" in payload.keys()
        assert "id" in payload.keys()
        assert record.partition == 1


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestNuclioMLRunJobs(tests.system.base.TestMLRunSystem):
    project_name = "nuclio-mlrun-jobs"

    def _deploy_function(self, replicas=1):
        filename = str(self.assets_path / "handler.py")
        fn = mlrun.code_to_function(
            filename=filename,
            name="nuclio-mlrun",
            kind="nuclio:mlrun",
            image="mlrun/mlrun",
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
