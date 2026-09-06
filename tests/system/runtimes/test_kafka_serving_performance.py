# Copyright 2026 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import os
import pathlib
import queue
import threading
import time
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import requests

import mlrun
import mlrun.errors
import tests.system.base

_WORKER_COUNT = int(os.getenv("MLRUN_SYSTEM_TESTS_KAFKA_WORKER_COUNT", "50"))
_REQUESTS_PER_WORKER = int(
    os.getenv("MLRUN_SYSTEM_TESTS_KAFKA_REQUESTS_PER_WORKER", "5000")
)
_EXPECTED_REQUEST_COUNT = _WORKER_COUNT * _REQUESTS_PER_WORKER
_PARTITION_COUNT = 3
_DEFAULT_KAFKA_PYTHON_VERSION = "2.3.2"
_DEFAULT_RESULTS_PATH = "/tmp/mlrun_kafka_performance_results.jsonl"
_DEFAULT_MONITOR_INTERVAL_SECONDS = 30.0
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
_DEFAULT_PERFORMANCE_TIMEOUT_SECONDS = 3 * 60 * 60
_MAX_FAILURE_EXAMPLES = 20


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.skipif(
    not os.getenv("MLRUN_SYSTEM_TESTS_KAFKA_BROKERS"),
    reason="MLRUN_SYSTEM_TESTS_KAFKA_BROKERS must be set",
)
@pytest.mark.skipif(
    os.getenv("MLRUN_SYSTEM_TESTS_RUN_KAFKA_PERFORMANCE", "").lower() != "true",
    reason="Set MLRUN_SYSTEM_TESTS_RUN_KAFKA_PERFORMANCE=true to run",
)
class TestKafkaServingPerformance(tests.system.base.TestMLRunSystem):
    """Opt-in ML-9935 Kafka serving performance reproducer."""

    env_file_path = pathlib.Path(
        os.getenv(
            "MLRUN_SYSTEM_TESTS_ENV_FILE",
            tests.system.base.TestMLRunSystem.env_file_path,
        )
    )
    project_name = os.getenv(
        "MLRUN_SYSTEM_TESTS_KAFKA_PROJECT",
        "kafka-serving-performance",
    )
    brokers = os.getenv("MLRUN_SYSTEM_TESTS_KAFKA_BROKERS")
    function_brokers = os.getenv(
        "MLRUN_SYSTEM_TESTS_KAFKA_FUNCTION_BROKERS",
        brokers,
    )
    kafka_python_version = os.getenv(
        "MLRUN_SYSTEM_TESTS_KAFKA_PYTHON_VERSION",
        _DEFAULT_KAFKA_PYTHON_VERSION,
    )
    run_id = uuid.uuid4().hex
    topic = f"mlrun-ml-9935-{run_id}"
    output_topic = f"mlrun-ml-9935-output-{run_id}"
    consumer_group = f"mlrun-ml-9935-consumer-{run_id}"
    image = os.getenv("MLRUN_SYSTEM_TESTS_KAFKA_FUNCTION_IMAGE", "mlrun/mlrun")

    @pytest.fixture()
    def kafka_clients(
        self,
    ) -> typing.Iterator[tuple[typing.Any, typing.Any]]:
        import kafka

        admin_client = kafka.KafkaAdminClient(bootstrap_servers=self.brokers)
        offsets_consumer = kafka.KafkaConsumer(
            bootstrap_servers=self.brokers,
            enable_auto_commit=False,
        )

        try:
            admin_client.create_topics(
                new_topics=[
                    kafka.admin.NewTopic(
                        name=self.topic,
                        num_partitions=_PARTITION_COUNT,
                        replication_factor=1,
                    ),
                    kafka.admin.NewTopic(
                        name=self.output_topic,
                        num_partitions=_PARTITION_COUNT,
                        replication_factor=1,
                    ),
                ]
            )
            yield admin_client, offsets_consumer
        finally:
            try:
                admin_client.delete_topics(topics=[self.topic, self.output_topic])
            except kafka.errors.KafkaError as exc:
                self._logger.warning(
                    "Failed deleting Kafka performance topic",
                    topic=self.topic,
                    error=mlrun.errors.err_to_str(exc),
                )
            try:
                admin_client.delete_consumer_groups(group_ids=[self.consumer_group])
            except kafka.errors.KafkaError as exc:
                self._logger.warning(
                    "Failed deleting Kafka performance consumer group",
                    consumer_group=self.consumer_group,
                    error=mlrun.errors.err_to_str(exc),
                )
            try:
                offsets_consumer.close()
            finally:
                admin_client.close()

    @pytest.mark.timeout(4 * 60 * 60)
    def test_kafka_queue_performance(
        self,
        kafka_clients: tuple[typing.Any, typing.Any],
    ) -> None:
        """Submit the ML-9935 workload and measure downstream zero-lag time."""
        admin_client, offsets_consumer = kafka_clients
        monitor_interval_seconds = float(
            os.getenv(
                "MLRUN_SYSTEM_TESTS_KAFKA_MONITOR_INTERVAL_SECONDS",
                str(_DEFAULT_MONITOR_INTERVAL_SECONDS),
            )
        )
        request_timeout_seconds = float(
            os.getenv(
                "MLRUN_SYSTEM_TESTS_KAFKA_REQUEST_TIMEOUT_SECONDS",
                str(_DEFAULT_REQUEST_TIMEOUT_SECONDS),
            )
        )
        performance_timeout_seconds = float(
            os.getenv(
                "MLRUN_SYSTEM_TESTS_KAFKA_PERFORMANCE_TIMEOUT_SECONDS",
                str(_DEFAULT_PERFORMANCE_TIMEOUT_SECONDS),
            )
        )
        results_path = pathlib.Path(
            os.getenv(
                "MLRUN_KAFKA_PERFORMANCE_RESULTS_PATH",
                _DEFAULT_RESULTS_PATH,
            )
        )
        result: dict[str, typing.Any] = {
            "consumer_group": self.consumer_group,
            "expected_request_count": _EXPECTED_REQUEST_COUNT,
            "function_brokers_differ_from_client": self.function_brokers
            != self.brokers,
            "kafka_python_version_requested": self.kafka_python_version,
            "partition_count": _PARTITION_COUNT,
            "requests_per_worker": _REQUESTS_PER_WORKER,
            "run_id": self.run_id,
            "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "failed",
            "topic": self.topic,
            "output_topic": self.output_topic,
            "worker_count": _WORKER_COUNT,
        }
        telemetry: list[dict[str, typing.Any]] = []
        failures = _FailureCollector()
        monitor_stop_event: threading.Event | None = None
        monitor_thread: threading.Thread | None = None

        try:
            function = self._deploy_performance_graph()

            workload_started = time.monotonic()
            deadline = workload_started + performance_timeout_seconds
            monitor_stop_event = threading.Event()
            completion_event = threading.Event()
            monitor_errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)
            monitor_thread = threading.Thread(
                target=self._monitor_offsets,
                kwargs={
                    "admin_client": admin_client,
                    "completion_event": completion_event,
                    "deadline": deadline,
                    "monitor_errors": monitor_errors,
                    "monitor_interval_seconds": monitor_interval_seconds,
                    "offsets_consumer": offsets_consumer,
                    "stop_event": monitor_stop_event,
                    "telemetry": telemetry,
                    "workload_started": workload_started,
                },
                name="kafka-performance-offset-monitor",
                daemon=True,
            )
            monitor_thread.start()

            submit_started = time.monotonic()
            attempted_request_count = self._submit_workload(
                failures=failures,
                request_timeout_seconds=request_timeout_seconds,
                url=function.get_url(),
            )
            result["submit_elapsed_seconds"] = time.monotonic() - submit_started
            result["attempted_request_count"] = attempted_request_count
            assert attempted_request_count == _EXPECTED_REQUEST_COUNT
            assert failures.count == 0, failures.summary

            completion_event.wait(
                timeout=max(0.0, deadline - time.monotonic()) + monitor_interval_seconds
            )
            monitor_stop_event.set()
            monitor_thread.join(timeout=monitor_interval_seconds + 10)
            assert not monitor_thread.is_alive(), "Kafka offset monitor did not stop"
            if not monitor_errors.empty():
                raise monitor_errors.get_nowait()
            assert completion_event.is_set(), (
                "Kafka consumer group did not commit all expected records with zero lag"
            )

            result["total_zero_lag_elapsed_seconds"] = telemetry[-1]["elapsed_seconds"]
            result["kafka_python_version_imported"] = self._read_imported_version()
            assert result["kafka_python_version_imported"] == self.kafka_python_version
            result["status"] = "passed"
        except BaseException as exc:
            result["error"] = mlrun.errors.err_to_str(exc)
            raise
        finally:
            if monitor_stop_event is not None:
                monitor_stop_event.set()
            if monitor_thread is not None and monitor_thread.is_alive():
                monitor_thread.join(timeout=monitor_interval_seconds + 10)
            result["completed_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            result["http_failure_count"] = failures.count
            result["http_failure_examples"] = failures.examples
            result["lag_telemetry"] = telemetry
            self._write_result(path=results_path, result=result)

    def _deploy_performance_graph(self) -> typing.Any:
        requirement = f"kafka-python=={self.kafka_python_version}"
        function = mlrun.code_to_function(
            name=f"kafka-performance-{self.run_id[:8]}",
            kind="serving",
            project=self.project_name,
            filename=str(self.assets_path / "kafka_performance.py"),
            image=self.image,
        )
        function.spec.build.commands = [
            f"python -m pip install --no-deps --force-reinstall {requirement}"
        ]
        function.spec.build.with_mlrun = False
        graph = function.set_topology("flow", engine="async")
        version_stamp = graph.to(
            name="version-stamp",
            class_name="KafkaVersionStamp",
            full_event=True,
        )
        version_stamp.to(
            ">>",
            "workload",
            path=f"kafka://{self.function_brokers}/{self.topic}",
            group=self.consumer_group,
            sharding_func=1,
            full_event=True,
        ).to(
            name="downstream",
            class_name="Identity",
            function="downstream",
        ).to(
            ">>",
            "output",
            path=self.output_topic,
            kafka_brokers=self.function_brokers,
            sharding_func=2,
        )
        graph.add_step(
            name="other-downstream",
            class_name="Augment",
            after="workload",
            function="other-downstream",
            full_event=True,
        )
        graph["output"].after_step("other-downstream")
        function.add_child_function(
            name="downstream",
            url=str(self.assets_path / "child_function.py"),
            image=self.image,
        )
        function.add_child_function(
            name="other-downstream",
            url=str(self.assets_path / "child_function.py"),
            image=self.image,
        )

        self._logger.info(
            "Deploying Kafka performance graph",
            consumer_group=self.consumer_group,
            kafka_python_version=self.kafka_python_version,
            topic=self.topic,
        )
        function.deploy()
        return function

    def _submit_workload(
        self,
        failures: "_FailureCollector",
        request_timeout_seconds: float,
        url: str,
    ) -> int:
        self._logger.info(
            "Submitting Kafka performance workload",
            request_count=_EXPECTED_REQUEST_COUNT,
            requests_per_worker=_REQUESTS_PER_WORKER,
            worker_count=_WORKER_COUNT,
        )
        with ThreadPoolExecutor(max_workers=_WORKER_COUNT) as executor:
            worker_futures = [
                executor.submit(
                    self._run_worker,
                    failures=failures,
                    request_timeout_seconds=request_timeout_seconds,
                    url=url,
                    worker_index=worker_index,
                )
                for worker_index in range(_WORKER_COUNT)
            ]
            return sum(future.result() for future in as_completed(worker_futures))

    def _run_worker(
        self,
        failures: "_FailureCollector",
        request_timeout_seconds: float,
        url: str,
        worker_index: int,
    ) -> int:
        session = requests.Session()
        try:
            for sequence in range(_REQUESTS_PER_WORKER):
                try:
                    response = session.post(
                        url=url,
                        json={"sequence": sequence, "worker": worker_index},
                        timeout=request_timeout_seconds,
                        verify=mlrun.mlconf.httpdb.http.verify,
                    )
                    if not response.ok:
                        failures.add(
                            {
                                "body": response.text[:500],
                                "sequence": sequence,
                                "status_code": response.status_code,
                                "worker": worker_index,
                            }
                        )
                except requests.RequestException as exc:
                    failures.add(
                        {
                            "error": mlrun.errors.err_to_str(exc),
                            "sequence": sequence,
                            "worker": worker_index,
                        }
                    )
        finally:
            session.close()
        return _REQUESTS_PER_WORKER

    def _read_imported_version(self) -> str:
        import kafka

        consumer = kafka.KafkaConsumer(
            self.output_topic,
            bootstrap_servers=self.brokers,
            auto_offset_reset="earliest",
            consumer_timeout_ms=60_000,
        )
        try:
            record = next(consumer)
            return json.loads(record.value)["kafka_version"]
        finally:
            consumer.close()

    def _monitor_offsets(
        self,
        admin_client: typing.Any,
        completion_event: threading.Event,
        deadline: float,
        monitor_errors: queue.Queue[BaseException],
        monitor_interval_seconds: float,
        offsets_consumer: typing.Any,
        stop_event: threading.Event,
        telemetry: list[dict[str, typing.Any]],
        workload_started: float,
    ) -> None:
        import kafka

        partitions = [
            kafka.TopicPartition(topic=self.topic, partition=partition)
            for partition in range(_PARTITION_COUNT)
        ]
        try:
            while not stop_event.is_set():
                end_offsets = offsets_consumer.end_offsets(partitions=partitions)
                committed_offsets = admin_client.list_consumer_group_offsets(
                    group_id=self.consumer_group
                )
                end_by_partition = {
                    str(partition.partition): end_offsets[partition]
                    for partition in partitions
                }
                committed_by_partition = {
                    str(partition.partition): max(
                        0,
                        committed_offsets[partition].offset
                        if partition in committed_offsets
                        else 0,
                    )
                    for partition in partitions
                }
                lag_by_partition = {
                    partition: max(
                        0,
                        end_by_partition[partition] - committed_by_partition[partition],
                    )
                    for partition in end_by_partition
                }
                sample = {
                    "committed_offsets": committed_by_partition,
                    "elapsed_seconds": time.monotonic() - workload_started,
                    "end_offsets": end_by_partition,
                    "lag": lag_by_partition,
                    "total_committed": sum(committed_by_partition.values()),
                    "total_end_offset": sum(end_by_partition.values()),
                    "total_lag": sum(lag_by_partition.values()),
                }
                telemetry.append(sample)
                self._logger.info("Kafka performance lag sample", **sample)

                if (
                    sample["total_end_offset"] == _EXPECTED_REQUEST_COUNT
                    and sample["total_committed"] == _EXPECTED_REQUEST_COUNT
                    and sample["total_lag"] == 0
                ):
                    completion_event.set()
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for the Kafka consumer group to reach zero lag"
                    )
                stop_event.wait(timeout=monitor_interval_seconds)
        # The monitor runs across a thread boundary, so every operational failure
        # must be returned to the test thread instead of being silently discarded.
        except Exception as exc:  # noqa: BLE001
            monitor_errors.put_nowait(exc)
            completion_event.set()

    def _write_result(
        self,
        path: pathlib.Path,
        result: dict[str, typing.Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="a", encoding="utf-8") as result_file:
            result_file.write(json.dumps(result, sort_keys=True))
            result_file.write("\n")
        self._logger.info(
            "Wrote Kafka performance result",
            path=str(path),
            status=result["status"],
        )


class _FailureCollector:
    def __init__(self) -> None:
        self._count = 0
        self._examples: list[dict[str, typing.Any]] = []
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def examples(self) -> list[dict[str, typing.Any]]:
        with self._lock:
            return list(self._examples)

    @property
    def summary(self) -> str:
        return (
            f"{self.count} HTTP requests failed; examples: "
            f"{json.dumps(self.examples, sort_keys=True)}"
        )

    def add(self, failure: dict[str, typing.Any]) -> None:
        with self._lock:
            self._count += 1
            if len(self._examples) < _MAX_FAILURE_EXAMPLES:
                self._examples.append(failure)
