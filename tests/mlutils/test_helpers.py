import unittest.mock

import mlrun
from mlrun.utils import logger


def test_create_linear_backoff():
    stop_value = 120
    base = 2
    coefficient = 4
    backoff = mlrun.utils.helpers.create_linear_backoff(base, coefficient, stop_value)
    for i in range(0, 120):
        expected_value = min(base + i * coefficient, stop_value)
        assert expected_value, next(backoff)


def test_create_linear_backoff_negative_coefficient():
    stop_value = 2
    base = 120
    coefficient = -4
    backoff = mlrun.utils.helpers.create_linear_backoff(base, coefficient, stop_value)
    for i in range(120, 0):
        expected_value = min(base + i * coefficient, stop_value)
        assert expected_value, next(backoff)


def test_create_exponential_backoff():
    base = 2
    max_value = 120
    backoff = mlrun.utils.helpers.create_exponential_backoff(base, max_value)
    for i in range(1, 120):
        expected_value = min(base ** i, max_value)
        assert expected_value, next(backoff)


def test_create_step_backoff():
    steps = [[2, 3], [10, 5], [120, None]]
    backoff = mlrun.utils.helpers.create_step_backoff(steps)
    for step in steps:
        step_value, step_occurrences = step
        if step_occurrences is not None:
            for _ in range(0, step_occurrences):
                assert step_value, next(backoff)
        else:

            # Run another 10 iterations:
            for _ in range(0, 10):
                assert step_value, next(backoff)


def test_retry_until_successful():
    def test_run(backoff):
        call_count = {"count": 0}
        unsuccessful_mock = unittest.mock.Mock()
        successful_mock = unittest.mock.Mock()

        def some_func(count_dict, a, b, some_other_thing=None):
            logger.debug(
                "Some function called", a=a, b=b, some_other_thing=some_other_thing
            )
            if count_dict["count"] < 3:
                logger.debug("Some function is still running, raising exception")
                count_dict["count"] += 1
                unsuccessful_mock()
                raise Exception("I'm running,try again later")

            logger.debug("Some function finished successfully")
            successful_mock()
            return "Finished"

        result = mlrun.utils.retry_until_successful(
            backoff,
            120,
            logger,
            True,
            some_func,
            call_count,
            5,
            [1, 8],
            some_other_thing="Just",
        )
        assert result, "Finished"
        assert unsuccessful_mock.call_count, 3
        assert successful_mock.call_count, 1

    test_run(0.02)

    test_run(mlrun.utils.create_linear_backoff(0.02, 0.02))
