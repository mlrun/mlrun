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

import os
import re
import time
from collections.abc import Callable
from typing import Any, Optional, TypeAlias

import pytest

import mlrun
import mlrun.common.model_monitoring.helpers
import mlrun.model_monitoring.helpers
from mlrun import MlrunProject
from mlrun.datastore.datastore_profile import (
    DatastoreProfile,
    DatastoreProfileKafkaSource,
    DatastoreProfileKafkaStream,
    DatastoreProfilePostgreSQL,
    DatastoreProfileTDEngine,
    DatastoreProfileV3io,
)
from tests.system.base import TestMLRunSystem

_ProfilesMap: TypeAlias = dict[str, type[DatastoreProfile]]

_DS_TYPE_TO_DS_PROFILE: _ProfilesMap = {
    "v3io": DatastoreProfileV3io,
    "taosws": DatastoreProfileTDEngine,
    "kafka_source": DatastoreProfileKafkaSource,
    "postgresql": DatastoreProfilePostgreSQL,
    "kafka_stream": DatastoreProfileKafkaStream,
}


@pytest.mark.model_monitoring
class TestMLRunSystemModelMonitoring(TestMLRunSystem):
    project: MlrunProject
    mm_tsdb_profile: DatastoreProfile
    mm_stream_profile: DatastoreProfile

    @staticmethod
    def _get_profile(profile_data: Any, profiles_map: _ProfilesMap) -> DatastoreProfile:
        if isinstance(profile_data, dict):
            ds_type = profile_data.get("type")
            if ds_type in profiles_map:
                return profiles_map[ds_type].parse_obj(profile_data)
            raise ValueError(
                f"Unsupported datastore type: '{ds_type}', expected one of {list(profiles_map)}"
            )
        raise ValueError("The model monitoring profile data is not a dictionary")

    @classmethod
    def get_tsdb_profile(cls, profile_data: dict[str, Any]) -> DatastoreProfile:
        return cls._get_profile(
            profile_data,
            {
                type_: _DS_TYPE_TO_DS_PROFILE[type_]
                for type_ in ("v3io", "taosws", "postgresql")
            },
        )

    @classmethod
    def get_stream_profile(cls, profile_data: dict[str, Any]) -> DatastoreProfile:
        profile = cls._get_profile(
            profile_data,
            {
                type_: _DS_TYPE_TO_DS_PROFILE[type_]
                for type_ in ("v3io", "kafka_source", "kafka_stream")
            },
        )
        if isinstance(profile, DatastoreProfileV3io):
            # Populate the V3IO access key for the stream profile
            profile.v3io_access_key = (
                profile.v3io_access_key or mlrun.mlconf.get_v3io_access_key()
            )
        return profile

    @classmethod
    def set_mm_profiles(cls):
        cls.mm_tsdb_profile = cls.get_tsdb_profile(cls.mm_tsdb_profile_data)
        cls.mm_stream_profile = cls.get_stream_profile(cls.mm_stream_profile_data)

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.set_mm_profiles()

    def set_mm_credentials(self) -> None:
        self.project.register_datastore_profile(self.mm_tsdb_profile)
        self.project.register_datastore_profile(self.mm_stream_profile)
        self.project.set_model_monitoring_credentials(
            tsdb_profile_name=self.mm_tsdb_profile.name,
            stream_profile_name=self.mm_stream_profile.name,
        )

    def get_stream_path(self, function_name) -> (str, str):
        """
        :returns: tuple of container and stream_path
        """
        stream_profile = TestMLRunSystemModelMonitoring.get_stream_profile(
            self.mm_stream_profile_data
        )
        stream_uri = mlrun.model_monitoring.helpers.get_stream_path(
            project=self.project.name,
            function_name=function_name,
            profile=stream_profile,
        )
        _, container, stream_path = (
            mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
                stream_uri,
            )
        )
        return container, stream_path

    @classmethod
    def wait_for_condition(
        cls,
        condition_check: Callable,
        initial_wait: float = 0.0,
        timeout: Optional[float] = None,
        retry_interval: float = 10.0,
        condition_description: str = "condition to be met",
    ) -> None:
        """Wait for a condition to be met by retrying until success or timeout.

        The condition_check function should use assertions to validate conditions.
        If it completes without raising an exception, the condition is considered met.
        If it raises an exception, the check will be retried until timeout.

        :param condition_check: Function that raises an exception if condition not met
        :param initial_wait: Time to wait before first check (seconds)
        :param timeout: Maximum time to wait (auto-calculated if not provided)
        :param retry_interval: Time between retry attempts (seconds)
        :param condition_description: Human-readable description for logging
        """
        # Auto-calculate timeout if not provided
        if timeout is None:
            timeout = max(initial_wait * 3, 60.0)  # At least 60s timeout

        if initial_wait > 0:
            cls._logger.debug(
                f"Initial wait before checking {condition_description}",
                wait_seconds=initial_wait,
            )
            time.sleep(initial_wait)

        start_time = time.time()
        attempt = 0

        while time.time() - start_time < timeout:
            attempt += 1
            elapsed = time.time() - start_time
            # Check if this is the last attempt (not enough time for another retry)
            last_check = elapsed + retry_interval >= timeout

            cls._logger.debug(
                f"Checking {condition_description}",
                attempt=attempt,
                elapsed_seconds=round(elapsed, 1),
                timeout_seconds=timeout,
                last_check=last_check,
            )

            try:
                condition_check()
                # No exception means condition is met
                cls._logger.info(
                    f"Condition met: {condition_description}",
                    attempt=attempt,
                    elapsed_seconds=round(elapsed, 1),
                )
                return
            except Exception:
                if last_check:
                    # On last attempt, let the actual exception propagate for better error reporting
                    raise
                # On earlier attempts, log and continue retrying
                cls._logger.debug(
                    "Exception during check, will retry",
                    attempt=attempt,
                    exc_info=True,
                )
                time.sleep(retry_interval)

        # Timeout reached without success
        elapsed = round(time.time() - start_time, 1)
        raise TimeoutError(
            f"Timeout after {elapsed}s waiting for {condition_description} "
            f"(timeout: {timeout}s, attempts: {attempt})"
        )

    def _is_kube_client_available(self) -> bool:
        """Check if kube_client is configured and available."""
        try:
            if not hasattr(self, "kube_client") or self.kube_client is None:
                return False
            # Test if it's a property that raises
            _ = self.kube_client.api_client
            return True
        except AttributeError:
            return False

    def _get_monitoring_pod_prefixes(self, project_name: str) -> list[str]:
        """Get pod prefixes for project-specific monitoring pods (full logs)."""
        return [
            f"nuclio-{project_name}-model-monitoring-stream",
            f"nuclio-{project_name}-model-monitoring-controller",
            f"nuclio-{project_name}-model-monitoring-writer",
            f"nuclio-{project_name}-model-serving",
        ]

    def _matches_prefix(self, pod_name: str, prefixes: list[str]) -> bool:
        """Check if pod_name matches any of the given prefixes."""
        return any(pod_name.startswith(prefix) for prefix in prefixes)

    def _collect_logs_from_pods_via_kube(
        self,
        namespace: str,
        project_name: str,
        monitoring_prefixes: list[str],
        api_prefixes: list[str],
        tail_lines: int,
    ) -> dict[str, str]:
        """Collect logs from pods using kube_client."""
        collected_logs = {}
        pods = self.kube_client.list_namespaced_pod(namespace)

        for pod in pods.items:
            pod_name = pod.metadata.name
            logs = self._collect_pod_logs(pod_name, namespace, tail_lines)
            if not logs:
                continue

            if self._matches_prefix(pod_name, monitoring_prefixes):
                collected_logs[pod_name] = logs
            elif self._matches_prefix(pod_name, api_prefixes):
                if filtered := self._filter_error_logs(logs, project_name):
                    collected_logs[f"{pod_name} (errors)"] = filtered

        return collected_logs

    def collect_monitoring_pod_logs(self, tail_lines: int = 500) -> dict[str, str]:
        """Collect logs from model monitoring related pods for debugging.

        :param tail_lines: Number of lines to retrieve from each pod's logs
        :returns: Dictionary mapping pod names to their logs
        """

        if not self._is_kube_client_available():
            self._logger.info(
                "kube_client not available, skipping pod log collection. "
                "Set MLRUN_SYSTEM_TEST_KUBECONFIG_PATH or MLRUN_SYSTEM_TEST_KUBECONFIG to enable."
            )
            return {}

        project_name = self.project_name
        monitoring_prefixes = self._get_monitoring_pod_prefixes(project_name)
        api_prefixes = ["mlrun-api-chief", "mlrun-api-worker"]

        try:
            return self._collect_logs_from_pods_via_kube(
                "default-tenant",
                project_name,
                monitoring_prefixes,
                api_prefixes,
                tail_lines,
            )
        except Exception as e:
            self._logger.warning("Failed to collect pod logs", error=str(e))
            return {}

    def _collect_pod_logs(
        self, pod_name: str, namespace: str, tail_lines: int
    ) -> Optional[str]:
        """Collect logs from a single pod."""
        try:
            logs = self.kube_client.read_namespaced_pod_log(
                name=pod_name, namespace=namespace, tail_lines=tail_lines
            )
            self._logger.info(f"Collected logs from {pod_name}", log_length=len(logs))
            return logs
        except Exception as e:
            self._logger.warning(
                f"Failed to collect logs from {pod_name}", error=str(e)
            )
            return f"Failed to get logs: {e}"

    def _filter_error_logs(
        self, logs: str, project_name: str, context_lines: int = 5
    ) -> str:
        """Filter logs to include warning/error lines mentioning project, with context.

        Captures N lines before each matching line to include call stacks/tracebacks.
        """
        lines = logs.splitlines()
        error_pattern = re.compile(
            r"(warning|warn|error|exception|traceback|failed|critical)",
            re.IGNORECASE,
        )

        matched_indices = set()
        for i, line in enumerate(lines):
            if error_pattern.search(line) and project_name in line:
                # Add context lines before the match
                for j in range(max(0, i - context_lines), i + 1):
                    matched_indices.add(j)

        return "\n".join(lines[i] for i in sorted(matched_indices))

    def print_monitoring_pod_logs(self, tail_lines: int = 500) -> None:
        """Log pod logs for CI visibility on test failure.

        :param tail_lines: Number of lines to retrieve from each pod's logs
        """
        logs = self.collect_monitoring_pod_logs(tail_lines=tail_lines)

        if not logs:
            self._logger.info("No monitoring pod logs collected")
            return

        self._logger.info("=== MODEL MONITORING POD LOGS (for debugging) ===")
        for pod_name, pod_logs in logs.items():
            self._logger.info(f"--- POD: {pod_name} ---\n{pod_logs}")
        self._logger.info("=== END OF MONITORING POD LOGS ===")
