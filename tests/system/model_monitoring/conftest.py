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

import pytest
from pytest import CallInfo, Function, TestReport


def _collect_monitoring_pod_logs(item: Function) -> None:
    """Collect and print pod logs for failed model monitoring tests.

    This helps debug test failures by showing the actual logs from monitoring
    pods (stream, controller, serving, mlrun-api/chief/worker) which may
    contain errors not visible in the test output.

    :param item: The pytest test item
    """
    try:
        test_instance = item.instance
        if test_instance is None:
            return

        if not hasattr(test_instance, "print_monitoring_pod_logs"):
            return

        print(f"\n>>> Collecting pod logs for FAILED test: {item.nodeid}")
        test_instance.print_monitoring_pod_logs(tail_lines=300)

    except Exception as exc:
        print(f">>> Failed to collect monitoring pod logs: {exc}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Function, call: CallInfo) -> TestReport:
    outcome = yield
    report: TestReport = outcome.get_result()

    # Collect pod logs only on test failure to help debug issues
    if report.when == "call" and not report.passed:
        _collect_monitoring_pod_logs(item)
