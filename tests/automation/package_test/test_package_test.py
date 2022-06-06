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
            if command == "pip install safety":
                return 0, "", ""
            else:
                if case.get("output_file"):
                    with open(case["output_file"]) as file:
                        output = file.readlines()
                        output = "".join(output)
                else:
                    output = case.get("output")
                code = 255 if output else 0
                return code, output, ""

        package_tester._run_command = unittest.mock.Mock(side_effect=_run_command_mock)
        if case.get("expected_to_fail"):
            with pytest.raises(AssertionError, match="Found vulnerable requirements"):
                package_tester._test_requirements_vulnerabilities("some-extra")
        else:
            package_tester._test_requirements_vulnerabilities("some-extra")
