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

import automation.package_test.test
import tests.conftest
from mlrun.utils import logger


@pytest.mark.parametrize(
    "case",
    [
        {
            "output": """
    {
        "vulnerabilities": [
            {
                "vulnerability_id": "44716",
                "package_name": "numpy",
                "vulnerable_spec": "<1.22.0",
                "all_vulnerable_specs": [
                    "<1.22.0"
                ],
                "analyzed_version": "1.21.6",
                "advisory": "Numpy 1.22.0 includes a fix for CVE-2021-41496",
                "CVE": "CVE-2021-41496",
                "severity": null,
                "affected_versions": [],
                "more_info_url": "https://pyup.io/v/44716/f17"
            }
        ]
    }""",
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
    ],
)
def test_requirements_vulnerabilities(case, monkeypatch):
    package_tester = automation.package_test.test.PackageTester(logger)

    logger.info("Testing case", case=case)

    def _run_command_mock(_, command, *args, **kwargs):
        # _test_requirements_vulnerabilities flow is running two commands:
        # 1. pip install safety - we don't care about it, so simply return success
        # 2. safety check --json - this is the actual one we want to mock the output for
        if "pip install safety" in command:
            return 0, "", ""
        elif "safety check --json" in command:
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

    monkeypatch.setattr(
        automation.package_test.test.Venv, "_run_command", _run_command_mock
    )
    monkeypatch.setattr(
        automation.package_test.test.Venv, "_create_venv", lambda _: None
    )
    monkeypatch.setattr(
        automation.package_test.test.Venv, "_clean_venv", lambda _: None
    )
    if case.get("expected_to_fail"):
        with pytest.raises(AssertionError, match="Found vulnerable requirements"):
            package_tester.test_requirements_vulnerabilities("some-extra")
    else:
        package_tester.test_requirements_vulnerabilities("some-extra")
