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


import collections
import os

import pytest
import requests
from _pytest.config import ExitCode
from _pytest.main import Session
from _pytest.python import Function
from _pytest.reports import TestReport
from _pytest.runner import CallInfo


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Function, call: CallInfo) -> TestReport:
    outcome = yield
    result: TestReport = outcome.get_result()

    try:
        post_report_failed_to_slack(
            result, os.getenv("MLRUN_SYSTEM_TESTS_SLACK_WEBHOOK_URL")
        )
    except Exception as exc:
        print(f"Failed to post test report to slack: {exc}")


def pytest_sessionstart(session):

    # caching test results
    session.results = collections.defaultdict(TestReport)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    # cache test call result
    if report.when == "call":
        item.session.results[item] = report


def pytest_sessionfinish(session: Session, exitstatus: ExitCode):
    slack_url = os.getenv("MLRUN_SYSTEM_TESTS_SLACK_WEBHOOK_URL")
    if slack_url:
        post_report_session_finish_to_slack(
            session,
            exitstatus,
            slack_url,
        )


def post_report_session_finish_to_slack(
    session: Session, exitstatus: ExitCode, slack_webhook_url
):
    mlrun_version = os.getenv("MLRUN_VERSION", "")
    mlrun_system_tests_component = os.getenv("MLRUN_SYSTEM_TESTS_COMPONENT", "")
    ttl_tests = session.testscollected
    ttl_failed_tests = session.testsfailed
    if exitstatus == ExitCode.OK:
        text = f"All {ttl_tests} tests passed successfully"
    else:
        text = f"{ttl_failed_tests} out of {ttl_tests} tests failed"

    test_session_info = ""
    if mlrun_system_tests_component:
        test_session_info += f"Component: {mlrun_system_tests_component}"
    else:
        test_session_info += f"Component: mlrun"

    # get mlrun version from envvar or fallback to git commit
    if mlrun_version:
        test_session_info += f"\nVersion: {mlrun_version}"
    else:
        try:
            git_commit = os.popen("git rev-parse --short HEAD").read().strip()
            test_session_info += f"\nVersion: {git_commit}"
        except Exception:
            test_session_info += f"\nVersion: unknown"

    test_session_info += f"\n{','.join(session.config.option.file_or_dir)}"
    text = f"*{text}*\n{test_session_info}"

    data = {"text": text, "attachments": []}
    for item, test_report in session.results.items():
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
