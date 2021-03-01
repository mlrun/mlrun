import pytest

import mlrun
import mlrun.errors


def test_with_limits_regex_validation():
    cases = [
        {"cpu": "no-number", "expected_failure": True},
        {"memory": "1g", "expected_failure": True},  # common mistake
        {"gpus": "1GPU", "expected_failure": True},
        {"cpu": "12e6"},
        {"memory": "12Mi"},
        {"memory": "12M"},
    ]
    for case in cases:
        function = mlrun.runtimes.KubejobRuntime()
        if case.get("expected_failure"):
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                function.with_limits(
                    case.get("memory"), case.get("cpu"), case.get("gpus")
                )
            with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
                function.with_requests(
                    case.get("memory"), case.get("cpu"), case.get("gpus")
                )
        else:
            function.with_limits(case.get("memory"), case.get("cpu"), case.get("gpus"))
            function.with_requests(
                case.get("memory"), case.get("cpu"), case.get("gpus")
            )
