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


import os

import pytest
import requests
from _pytest.python import Function
from _pytest.reports import TestReport
from _pytest.runner import CallInfo


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Function, call: CallInfo) -> TestReport:
    outcome = yield
    result: TestReport = outcome.get_result()

    try:
        post_report_to_slack(result, os.getenv("MLRUN_TEST_SLACK_WEBHOOK_URL"))
    except Exception as exc:
        print(f"Failed to post test report to slack: {exc}")


def post_report_to_slack(report: TestReport, slack_webhook_url: str):
    if not report.failed:
        return

    if not slack_webhook_url:
        return

    data = {
        "text": f"Test `{report.head_line}` has failed.",
        "attachments": [
            {
                "color": "danger",
                "title": "System test failure details",
                "fields": [
                    {
                        "title": "Failure",
                        "value": f"```{report.longreprtext}```",
                        "short": False,
                    },
                ],
            },
        ],
    }
    res = requests.post(slack_webhook_url, json=data)
    res.raise_for_status()
