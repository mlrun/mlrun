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
#
import base64
import collections
import os
import time
from tempfile import NamedTemporaryFile
from typing import Self

import humanfriendly
import kubernetes.client as k8s_client
import pytest
import requests
from _pytest.terminal import TerminalReporter
from kubernetes import config
from pytest import CallInfo, ExitCode, Function, TestReport

from mlrun.utils import create_test_logger

logger = create_test_logger(name="test-system")


def pytest_sessionstart(
    session: pytest.Session,
):
    setup_k8s_client(
        session=session,
    )  # Setup K8S client for use in system tests.

    # caching test results
    session.results = collections.defaultdict(TestReport)


def setup_k8s_client(
    session: pytest.Session,
):
    kubeconfig_content = None
    try:
        base64_kubeconfig_content = os.environ["SYSTEM_TEST_KUBECONFIG"]
        kubeconfig_content = base64.b64decode(base64_kubeconfig_content)
    except (ValueError, KeyError) as exc:
        logger.warning("Kubeconfig was empty or invalid.", exc_info=exc)
        session.kube_client = property(missing_kubeclient)
    if kubeconfig_content:
        with NamedTemporaryFile() as tempfile:
            tempfile.write(kubeconfig_content)
            tempfile.flush()
            try:
                config.load_kube_config(
                    config_file=tempfile.name,
                )
                session.kube_client = k8s_client.CoreV1Api()
            except config.config_exception.ConfigException:
                logger.warning(
                    "Failed to load kubeconfig, kube_client will be unavailable."
                )
                session.kube_client = property(missing_kubeclient)
    else:
        session.kube_client = property(missing_kubeclient)


def missing_kubeclient(self: Self):
    raise AttributeError("Kubeclient was not setup and is unavailable")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Function, call: CallInfo) -> TestReport:
    outcome = yield
    report: TestReport = outcome.get_result()

    # cache test call result
    if report.when == "call":
        item.session.results[item] = report

    # commented due to spamming
    # try:
    #     post_report_failed_to_slack(
    #         report, os.getenv("MLRUN_SYSTEM_TESTS_SLACK_WEBHOOK_URL")
    #     )
    # except Exception as exc:
    #     print(f"Failed to post test report to slack: {exc}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: ExitCode):
    slack_url = os.getenv("MLRUN_SYSTEM_TESTS_SLACK_WEBHOOK_URL")
    if slack_url:
        post_report_session_finish_to_slack(
            session,
            exitstatus,
            slack_url,
        )


def post_report_session_finish_to_slack(
    session: pytest.Session, exitstatus: ExitCode, slack_webhook_url
):
    reporter: TerminalReporter = session.config.pluginmanager.get_plugin(
        "terminalreporter"
    )
    test_duration = time.time() - reporter._sessionstarttime
    mlrun_version = os.getenv("MLRUN_VERSION", "")
    mlrun_current_branch = os.getenv("MLRUN_SYSTEM_TESTS_BRANCH", "")
    mlrun_system_tests_component = os.getenv("MLRUN_SYSTEM_TESTS_COMPONENT", "")
    run_url = os.getenv(
        "MLRUN_SYSTEM_TESTS_GITHUB_RUN_URL",
        "https://github.com/mlrun/mlrun/actions/workflows/system-tests-enterprise.yml",
    )
    total_executed_tests = session.testscollected
    total_failed_tests = session.testsfailed
    text = ""
    if mlrun_current_branch:
        text += f"[{mlrun_current_branch}] "

    if exitstatus == ExitCode.OK:
        text += f"All {total_executed_tests} tests passed successfully"
    else:
        text += f"{total_failed_tests} out of {total_executed_tests} tests failed"

    test_session_info = ""
    if mlrun_system_tests_component:
        test_session_info += f"Component: {mlrun_system_tests_component}"
    else:
        test_session_info += "Component: mlrun"

    # get mlrun version from envvar or fallback to git commit
    if mlrun_version:
        test_session_info += f"\nVersion: {mlrun_version}"
    else:
        try:
            git_commit = os.popen("git rev-parse --short HEAD").read().strip()
            test_session_info += f"\nVersion: {git_commit}"
        except Exception:
            test_session_info += "\nVersion: unknown"

    test_session_info += f"\nDuration: {humanfriendly.format_timespan(test_duration)}"
    test_session_info += f"\nRun URL: {run_url}"
    test_session_info += f"\nPath: {','.join(session.config.option.file_or_dir)}"
    text = f"*{text}*\n{test_session_info}"

    data = {"text": text, "attachments": []}
    for item, test_report in session.results.items():
        # currently do not send information about passed tests
        # it spams the slack channel and makes message big enough for API to allow it
        if test_report.passed:
            continue

        item: Function = item
        test_report: TestReport = test_report
        test_failed = test_report.failed
        data["attachments"].append(
            {
                "color": "danger" if test_failed else "good",
                "fields": [
                    {
                        "value": f"{item.nodeid} - {'Failed' if test_failed else 'Passed'}",
                        "short": False,
                    }
                ],
            }
        )

    # attach up to 10 items
    data["attachments"] = data["attachments"][:10]
    res = requests.post(slack_webhook_url, json=data)
    res.raise_for_status()


def post_report_failed_to_slack(report: TestReport, slack_webhook_url: str):
    if not report.failed:
        return

    if not slack_webhook_url:
        return

    data = {
        "text": f"Test `{report.head_line}` has failed.",
        "attachments": [
            {
                "color": "danger",
                "fields": [
                    {
                        "value": f"```{report.longreprtext[-1000:]}```",
                        "short": False,
                    },
                ],
            },
        ],
    }
    res = requests.post(slack_webhook_url, json=data)
    res.raise_for_status()
