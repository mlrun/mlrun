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
import unittest.mock

import pytest

import automation.package_test.test
import tests.conftest
from mlrun.utils import logger


def test_test_requirements_vulnerabilities():
    package_tester = automation.package_test.test.PackageTester()
    cases = [
        {
            "output": """
[
    [
        "fastapi",
        "<0.75.2",
        "0.67.0",
        "Fastapi 0.75.2 updates its dependency 'ujson' ranges to include a security fix.",
        "48159",
        null,
        null
    ]
]""",
            "expected_to_fail": True,
        },
        {
            "output_file": tests.conftest.tests_root_directory
            / "automation"
            / "package_test"
            / "assets"
            / "ignored_vulnerabilities.json",
        },
        {
            "output": "",
        },
    ]
    for case in cases:
        logger.info("Testing case", case=case)

        def _run_command_mock(command, *args, **kwargs):
            # _test_requirements_vulnerabilities flow is running two commands:
            # 1. pip install safety - we don't care about it, so simply return success
            # 2. safety check --json - this is the actual one we want to mock the output for
            if command == "pip install safety":
                return 0, "", ""
            elif command == "safety check --json":
                if case.get("output_file"):
                    with open(case["output_file"]) as file:
                        output = file.readlines()
                        output = "".join(output)
                else:
                    output = case.get("output")
                code = 255 if output else 0
                return code, output, ""
            else:
                raise NotImplementedError(f"Got unexpected command: {command}")

        package_tester._run_command = unittest.mock.Mock(side_effect=_run_command_mock)
        if case.get("expected_to_fail"):
            with pytest.raises(AssertionError, match="Found vulnerable requirements"):
                package_tester._test_requirements_vulnerabilities("some-extra")
        else:
            package_tester._test_requirements_vulnerabilities("some-extra")
