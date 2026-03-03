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

import base64
import json
import os
import pathlib
import re
import sys
import typing
import urllib.parse
from tempfile import NamedTemporaryFile

import git
import igz_mgmt
import kubernetes.client as k8s_client
import kubernetes.config
import pytest
import yaml
from deepdiff import DeepDiff

import mlrun.common.schemas
from mlrun import get_run_db, mlconf
from mlrun.utils import create_test_logger

logger = create_test_logger(name="test-system")


class TestMLRunSystem:
    project_name = "system-test-project"
    root_path = pathlib.Path(__file__).absolute().parent.parent.parent
    env_file_path = root_path / "tests" / "system" / "env.yml"
    results_path = root_path / "tests" / "test_results" / "system"
    enterprise_marker_name = "enterprise"
    model_monitoring_marker_name = "model_monitoring"
    model_monitoring_marker = False
    mandatory_env_vars = [
        "MLRUN_DBPATH",
    ]
    mandatory_enterprise_env_vars = mandatory_env_vars + [
        "V3IO_API",
        "V3IO_FRAMESD",
        "V3IO_USERNAME",
        "V3IO_ACCESS_KEY",
        "MLRUN_IGUAZIO_API_URL",
        "MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE",
    ]

    model_monitoring_mandatory_keys = [
        "mlrun_model_monitoring_tsdb_profile",
        "mlrun_model_monitoring_stream_profile",
    ]

    enterprise_configured = os.getenv("V3IO_API")

    _logger = logger

    _test_env = {}
    _old_env = {}

    @classmethod
    def setup_class(cls):
        env = cls._get_env_from_file()
        cls._setup_env(env)
        cls._setup_k8s_client()
        cls._run_db = get_run_db()
        cls.custom_setup_class()
        cls._logger = logger.get_child(cls.__name__.lower())
        cls.project: mlrun.projects.MlrunProject | None = None

        cls.mm_tsdb_profile_data = cls._get_mm_data(
            env, "mlrun_model_monitoring_tsdb_profile"
        )
        cls.mm_stream_profile_data = cls._get_mm_data(
            env, "mlrun_model_monitoring_stream_profile"
        )

        cls.uploaded_code = False

        if "MLRUN_IGUAZIO_API_URL" in env:
            cls._igz_mgmt_client = igz_mgmt.Client(
                endpoint=env["MLRUN_IGUAZIO_API_URL"],
                access_key=env["V3IO_ACCESS_KEY"],
            )

        # the dbpath is already configured on the test startup before this stage
        # so even though we set the env var, we still need to directly configure
        # it in mlconf.
        mlconf.dbpath = cls._test_env["MLRUN_DBPATH"]

    @staticmethod
    def _get_mm_data(
        env: dict[str, typing.Any], key: str
    ) -> dict[str, typing.Any] | None:
        data = env.get(key)
        if isinstance(data, str):
            data = json.loads(data)
        return data

    @classmethod
    def custom_setup_class(cls):
        pass

    def setup_method(self, method):
        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )

        self._setup_env(self._get_env_from_file())
        self._run_db = get_run_db()
        self.remote_code_dir = mlrun.utils.helpers.template_artifact_path(
            mlrun.mlconf.artifact_path, self.project_name
        )
        self._files_to_upload = []

        if not self._skip_set_environment():
            self.project = mlrun.get_or_create_project(
                self.project_name, "./", allow_cross_project=True
            )

        self.custom_setup()

        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    @staticmethod
    def _should_clean_resources():
        return os.environ.get("MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES") != "false"

    def _delete_test_project(self, name=None):
        if self._should_clean_resources():
            self._run_db.delete_project(
                name or self.project_name,
                deletion_strategy=mlrun.common.schemas.DeletionStrategy.cascading,
            )

    def teardown_method(self, method):
        self._logger.info(
            f"Tearing down test {self.__class__.__name__}::{method.__name__}"
        )

        self._logger.debug("Removing test data from database")
        if self._should_clean_resources():
            fsets = self._run_db.list_feature_sets()
            if fsets:
                for fset in fsets:
                    fset.purge_targets()

        self._delete_test_project()

        self.custom_teardown()

        self._logger.info(
            f"Finished tearing down test {self.__class__.__name__}::{method.__name__}"
        )

    @classmethod
    def teardown_class(cls):
        cls.custom_teardown_class()
        cls._teardown_env()

    def custom_setup(self):
        pass

    def custom_teardown(self):
        pass

    @classmethod
    def custom_teardown_class(cls):
        pass

    @staticmethod
    def _skip_set_environment():
        return False

    @classmethod
    def skip_test_if_env_not_configured(cls, test):
        mandatory_env_vars = (
            cls.mandatory_enterprise_env_vars
            if cls._has_marker(test, cls.enterprise_marker_name)
            else cls.mandatory_env_vars
        )
        if cls._has_marker(test, cls.model_monitoring_marker_name):
            # Use + (not +=) to avoid mutating the class variable in-place,
            # which would permanently append to it across test runs in the same process.
            mandatory_env_vars = (
                mandatory_env_vars + cls.model_monitoring_mandatory_keys
            )

        missing_env_vars = []
        try:
            env = cls._get_env_from_file()
        except FileNotFoundError:
            missing_env_vars = mandatory_env_vars
        else:
            for env_var in mandatory_env_vars:
                if env_var not in env or env[env_var] is None:
                    missing_env_vars.append(env_var)

        return pytest.mark.skipif(
            len(missing_env_vars) > 0,
            reason=f"This is a system test, add the needed environment variables {(*mandatory_env_vars,)} "
            f"in tests/system/env.yml. You are missing: {missing_env_vars}",
        )(test)

    @classmethod
    def is_enterprise_environment(cls):
        try:
            env = cls._get_env_from_file()
        except FileNotFoundError:
            return False
        else:
            for env_var in cls.mandatory_enterprise_env_vars:
                if env_var not in env or env[env_var] is None:
                    return False
            return True

    @classmethod
    def get_assets_path(cls):
        return (
            pathlib.Path(sys.modules[cls.__module__].__file__).absolute().parent
            / "assets"
        )

    @property
    def assets_path(self) -> pathlib.Path:
        """Returns the test file directory "assets" directory."""
        return self.get_assets_path()

    @classmethod
    def _get_env_from_file(cls) -> dict:
        with cls.env_file_path.open() as f:
            return yaml.safe_load(f)

    @classmethod
    def _setup_env(cls, env: dict):
        cls._logger.debug("Setting up test environment")
        cls._test_env.update(env)

        # Process keys
        for key, value in env.items():
            if key in cls.model_monitoring_mandatory_keys:
                # model monitoring profiles data is saved separately
                continue
            cls._process_env_var(key, value)

        # Reload the config so changes to the env vars will take effect
        mlrun.mlconf.reload()

    @classmethod
    def _process_env_var(cls, key, value):
        if key in os.environ:
            # Save old env vars for returning them on teardown
            cls._old_env[key] = os.environ[key]

        # Set the environment variable
        if isinstance(value, bool):
            os.environ[key] = "true" if value else "false"
        elif value is not None and not isinstance(value, list | dict):
            os.environ[key] = value

    @classmethod
    def _setup_k8s_client(cls):
        def missing_kubeclient(*args, **kwargs):
            raise AttributeError("Kubeclient was not setup and is unavailable")

        kubeconfig_content = None
        try:
            if kubeconfig_path := os.environ.get("MLRUN_SYSTEM_TEST_KUBECONFIG_PATH"):
                with open(kubeconfig_path, "rb") as file:
                    kubeconfig_content = file.read()
            elif base64_kubeconfig_content := os.environ.get(
                "MLRUN_SYSTEM_TEST_KUBECONFIG"
            ):
                kubeconfig_content = base64.b64decode(base64_kubeconfig_content)
        except ValueError as exc:
            logger.warning(
                "Kubeconfig was empty or invalid.",
                exc_info=mlrun.errors.err_to_str(exc),
            )
            cls.kube_client = property(missing_kubeclient)
        if kubeconfig_content:
            with NamedTemporaryFile() as tempfile:
                tempfile.write(kubeconfig_content)
                tempfile.flush()
                try:
                    kubernetes.config.load_kube_config(
                        config_file=tempfile.name,
                    )
                    cls.kube_client = k8s_client.CoreV1Api()
                except kubernetes.config.config_exception.ConfigException:
                    logger.warning(
                        "Failed to load kubeconfig, kube_client will be unavailable."
                    )
                    cls.kube_client = property(missing_kubeclient)
        else:
            cls.kube_client = property(missing_kubeclient)

    @classmethod
    def _teardown_env(cls):
        cls._logger.debug("Tearing down test environment")
        for env_var in cls._test_env:
            if env_var in os.environ:
                del os.environ[env_var]
        os.environ.update(cls._old_env)
        # reload the config so changes to the env vars will take affect
        mlrun.mlconf.reload()

    def _get_v3io_user_store_path(self, path: pathlib.Path, remote: bool = True) -> str:
        v3io_user = self._test_env["V3IO_USERNAME"]
        prefixes = {
            "remote": f"v3io:///users/{v3io_user}",
            "local": "/User",
        }
        prefix = prefixes["remote"] if remote else prefixes["local"]
        return prefix + str(path)

    def _verify_run_spec(
        self,
        run_spec,
        parameters: dict | None = None,
        inputs: dict | None = None,
        outputs: list | None = None,
        output_path: str | None = None,
        function: str | None = None,
        secret_sources: list | None = None,
        data_stores: list | None = None,
        scrape_metrics: bool | None = None,
    ):
        self._logger.debug("Verifying run spec", spec=run_spec)
        if parameters:
            self._assert_with_deepdiff(parameters, run_spec["parameters"])
        if inputs:
            self._assert_with_deepdiff(inputs, run_spec["inputs"])
        if outputs:
            self._assert_with_deepdiff(outputs, run_spec["outputs"])
        if output_path:
            assert run_spec["output_path"] == output_path
        if function:
            self._assert_with_deepdiff(function, run_spec["function"])
        if secret_sources:
            self._assert_with_deepdiff(secret_sources, run_spec["secret_sources"])
        if data_stores:
            self._assert_with_deepdiff(data_stores, run_spec["data_stores"])
        if scrape_metrics is not None:
            assert run_spec["scrape_metrics"] == scrape_metrics

    def _verify_run_metadata(
        self,
        run_metadata,
        uid: str | None = None,
        name: str | None = None,
        project: str | None = None,
        labels: dict | None = None,
        iteration: int | None = None,
    ):
        self._logger.debug("Verifying run metadata", spec=run_metadata)
        if uid:
            assert run_metadata["uid"] == uid
        if name:
            assert run_metadata["name"] == name
        if project:
            assert run_metadata["project"] == project
        if iteration:
            assert run_metadata["iteration"] == project
        if labels:
            for label, label_value in labels.items():
                assert label in run_metadata["labels"]
                assert run_metadata["labels"][label] == label_value

    def _verify_run_outputs(
        self,
        run_outputs,
        uid: str,
        name: str,
        project: str,
        output_path: pathlib.Path,
        accuracy: int | None = None,
        loss: int | None = None,
        best_iteration: int | None = None,
        iteration_results: bool = False,
        iteration: int | None = None,
    ):
        fragment = "" if iteration is None else f"#{iteration}"

        self._logger.debug("Verifying run outputs", spec=run_outputs)
        assert run_outputs["plotly"].startswith(str(output_path))
        assert (
            f"store://datasets/{project}/{name}_mydf{fragment}:latest@{uid}"
            in run_outputs["mydf"]
        )
        assert (
            f"store://artifacts/{project}/{name}_model{fragment}:latest@{uid}"
            in run_outputs["model"]
        )
        assert (
            f"store://artifacts/{project}/{name}_html_result{fragment}:latest@{uid}"
            in run_outputs["html_result"]
        )
        if accuracy:
            assert run_outputs["accuracy"] == accuracy
        if loss:
            assert run_outputs["loss"] == loss
        if best_iteration:
            assert run_outputs["best_iteration"] == best_iteration
        if iteration_results:
            assert run_outputs["iteration_results"].startswith(str(output_path))

    @staticmethod
    def _has_marker(test: typing.Callable, marker_name: str) -> bool:
        try:
            return (
                len([mark for mark in test.pytestmark if mark.name == marker_name]) > 0
            )
        except AttributeError:
            return False

    @staticmethod
    def _assert_with_deepdiff(expected, actual, ignore_order=True):
        if ignore_order:
            assert DeepDiff(expected, actual, ignore_order=True) == {}
        else:
            assert expected == actual

    def _upload_code_to_cluster(self):
        if not self.uploaded_code:
            for file in self._files_to_upload:
                source_path = str(self.assets_path / file)
                mlrun.get_dataitem(os.path.join(self.remote_code_dir, file)).upload(
                    source_path
                )
        self.uploaded_code = True

    @staticmethod
    def _resolve_current_git_branch_and_fork():
        """
        Resolve the current git branch and fork name.
        Falls back to any available remote if 'origin' is not found.
        """
        repo = git.Repo(search_parent_directories=True)

        # Try to get the 'origin' remote, or fall back to the first available remote
        remote = (
            repo.remotes.origin
            if "origin" in repo.remotes
            else next(iter(repo.remotes), None)
        )
        if remote is None:
            raise RuntimeError("No remotes found in the Git repository.")

        git_url = remote.url
        fork = TestMLRunSystem._extract_fork(git_url)
        branch = repo.active_branch.name

        return branch, fork

    @staticmethod
    def _extract_fork(git_url: str) -> str:
        """
        Return the user / organisation part (“fork”) from common Git remote URLs.
        Supports:
          • git@github.com:<fork>/<repo>.git      (classic SSH / scp-like)
          • https://github.com/<fork>/<repo>.git  (HTTPS)
          • ssh://git@github.com/<fork>/<repo>.git
          • git://github.com/<fork>/<repo>.git
        """
        # 1) scp-like SSH form: git@github.com:fork/repo(.git)
        match = re.match(r"git@[^:]+:([^/]+)/", git_url)
        if match:
            return match.group(1)

        # 2) Anything with “://” – let urlparse do the heavy lifting
        if "://" in git_url:
            parsed = urllib.parse.urlparse(git_url)
            # parsed.path -> "/fork/repo.git"; we only need the first component
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2:
                return parts[0]

        raise ValueError(f"Could not extract fork from git URL: {git_url}")

    # =========================================================================
    # Pod Log Collection for Test Failure Debugging
    # =========================================================================

    # System pod prefixes - these are shared pods, not project-specific
    SYSTEM_POD_PREFIXES = ("mlrun-api-chief", "mlrun-api-worker")

    # Default namespace for MLRun pods
    DEFAULT_NAMESPACE = "default-tenant"

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

    def _is_project_pod(self, pod_name: str, project_name: str) -> bool:
        """Check if pod belongs to the test project (name contains project name)."""
        return project_name in pod_name

    def _is_system_pod(self, pod_name: str) -> bool:
        """Check if pod is a system pod (mlrun-api-*)."""
        return pod_name.startswith(self.SYSTEM_POD_PREFIXES)

    def _collect_single_pod_logs(
        self,
        pod_name: str,
        namespace: str,
        tail_lines: int,
        since_seconds: int | None = None,
    ) -> str | None:
        """Collect logs from a single pod.

        :param pod_name: Name of the pod
        :param namespace: Kubernetes namespace
        :param tail_lines: Maximum number of lines to retrieve
        :param since_seconds: Only return logs newer than this many seconds
        :returns: Pod logs or error message
        """
        try:
            logs = self.kube_client.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=tail_lines,
                since_seconds=since_seconds,
            )
            self._logger.debug(
                f"Collected logs from {pod_name}",
                lines=len(logs.splitlines()) if logs else 0,
            )
            return logs
        except Exception as e:
            self._logger.warning(f"Failed to collect logs from {pod_name}: {e}")
            return f"[Failed to get logs: {e}]"

    def collect_pod_logs_on_failure(
        self,
        test_duration_seconds: int,
        tail_lines: int = 1000,
        time_buffer_seconds: int = 60,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> dict[str, str]:
        """Collect logs from relevant pods for debugging test failures.

        Collects logs from:
        - Project pods (name contains project_name): full logs (tail_lines)
        - System pods (mlrun-api-*): time-bounded logs (since_seconds)

        :param test_duration_seconds: How long the test ran (for since_seconds calc)
        :param tail_lines: Maximum lines per pod (default 1000)
        :param time_buffer_seconds: Extra seconds to add to since_seconds (default 60)
        :param namespace: Kubernetes namespace (default: default-tenant)
        :returns: Dictionary mapping pod names to their logs
        """
        if not self._is_kube_client_available():
            self._logger.info(
                "kube_client not available, skipping pod log collection. "
                "Set MLRUN_SYSTEM_TEST_KUBECONFIG_PATH or MLRUN_SYSTEM_TEST_KUBECONFIG."
            )
            return {}

        project_name = self.project_name
        since_seconds = test_duration_seconds + time_buffer_seconds
        collected_logs = {}

        try:
            pods = self.kube_client.list_namespaced_pod(namespace)
        except Exception as e:
            # EKS exec tokens (aws eks get-token / aws-iam-authenticator) are short lived (~15 minutes).
            # The kube client can hold a token loaded earlier in the test run, and long running tests can fail
            # after it expires, causing 401 Unauthorized on log collection. If we detect 401, refresh kubeconfig
            # (re-run exec) and retry once.
            status = getattr(e, "status", None)
            if status == 401 or "Unauthorized" in str(e) or "(401)" in str(e):
                self._logger.info(
                    f"Unauthorized listing pods in {namespace}, refreshing kube client and retrying once"
                )
                try:
                    type(self)._setup_k8s_client()
                    pods = self.kube_client.list_namespaced_pod(namespace)
                except Exception as e2:
                    self._logger.warning(f"Failed to list pods in {namespace}: {e2}")
                    return {}
            else:
                self._logger.warning(f"Failed to list pods in {namespace}: {e}")
                return {}

        for pod in pods.items:
            pod_name = pod.metadata.name

            if self._is_project_pod(pod_name, project_name):
                if logs := self._collect_single_pod_logs(
                    pod_name, namespace, tail_lines, since_seconds=None
                ):
                    collected_logs[pod_name] = logs

            elif self._is_system_pod(pod_name):
                if logs := self._collect_single_pod_logs(
                    pod_name, namespace, tail_lines, since_seconds=since_seconds
                ):
                    collected_logs[f"{pod_name} (last {since_seconds}s)"] = logs

        return collected_logs

    def print_pod_logs(self, logs: dict[str, str]) -> None:
        """Print collected pod logs for CI visibility.

        :param logs: Dictionary mapping pod names to their logs
        """
        if not logs:
            self._logger.info("No pod logs collected")
            return

        self._logger.info("=" * 60)
        self._logger.info("POD LOGS FOR DEBUGGING TEST FAILURE")
        self._logger.info("=" * 60)

        for pod_name, pod_logs in logs.items():
            self._logger.info(f"\n--- {pod_name} ---")
            # Print directly to ensure it appears in CI output
            print(pod_logs)

        self._logger.info("=" * 60)
        self._logger.info("END OF POD LOGS")
        self._logger.info("=" * 60)
