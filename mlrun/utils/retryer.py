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

import asyncio
import time

import mlrun.errors


def create_linear_backoff(base=2, coefficient=2, stop_value=120):
    """
    Create a generator of linear backoff. Check out usage example in test_helpers.py
    """
    x = 0
    comparison = min if coefficient >= 0 else max

    while True:
        next_value = comparison(base + x * coefficient, stop_value)
        yield next_value
        x += 1


def create_step_backoff(steps=None):
    """
    Create a generator of steps backoff.
    Example: steps = [[2, 5], [20, 10], [120, None]] will produce a generator in which the first 5
    values will be 2, the next 10 values will be 20 and the rest will be 120.
    :param steps: a list of lists [step_value, number_of_iteration_in_this_step]
    """
    steps = steps if steps is not None else [[2, 10], [10, 10], [120, None]]
    steps = iter(steps)

    # Get first step
    step = next(steps)
    while True:
        current_step_value, current_step_remain = step
        if current_step_remain == 0:
            # No more in this step, moving on
            step = next(steps)
        elif current_step_remain is None:
            # We are in the last step, staying here forever
            yield current_step_value
        elif current_step_remain > 0:
            # Still more remains in this step, just reduce the remaining number
            step[1] -= 1
            yield current_step_value


def create_exponential_backoff(base=2, max_value=120, scale_factor=1):
    """
    Create a generator of exponential backoff. Check out usage example in test_helpers.py
    :param base: exponent base
    :param max_value: max limit on the result
    :param scale_factor: factor to be used as linear scaling coefficient
    """
    exponent = 1
    while True:
        # This "complex" implementation (unlike the one in linear backoff) is to avoid exponent growing too fast and
        # risking going behind max_int
        next_value = scale_factor * (base**exponent)
        if next_value < max_value:
            exponent += 1
            yield next_value
        else:
            yield max_value


class Retryer:
    def __init__(self, backoff, timeout, logger, verbose, function, *args, **kwargs):
        """
        Initialize function retryer with given *args and **kwargs.
        Tries to run it until success or timeout reached (timeout is optional)
        :param backoff: can either be a:
                - number (int / float) that will be used as interval.
                - generator of waiting intervals. (support next())
        :param timeout: pass None if timeout is not wanted, number of seconds if it is
        :param logger: a logger so we can log the failures
        :param verbose: whether to log the failure on each retry
        :param _function: function to run
        :param args: functions args
        :param kwargs: functions kwargs
        """
        self.backoff = backoff
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start_time = None
        self.last_exception = None
        self.first_interval = None

    def run(self):
        self._prepare()
        while not self._timeout_exceeded():
            next_interval = self.first_interval or next(self.backoff)
            result, exc, retry = self._perform_call(next_interval)
            if retry:
                time.sleep(next_interval)
            elif not exc:
                return result
            else:
                break

        self._raise_last_exception()

    def _prepare(self):
        self.start_time = time.monotonic()
        self.last_exception = None

        # Check if backoff is just a simple interval
        if isinstance(self.backoff, int) or isinstance(self.backoff, float):
            self.backoff = create_linear_backoff(base=self.backoff, coefficient=0)

        self.first_interval = next(self.backoff)
        if self.timeout and self.timeout <= self.first_interval:
            self.logger.warning(
                f"Timeout ({self.timeout}) must be higher than backoff ({self.first_interval})."
                f" Set timeout to be higher than backoff."
            )

    def _perform_call(self, next_interval):
        try:
            result = self.function(*self.args, **self.kwargs)
            return result, None, False
        except mlrun.errors.MLRunFatalFailureError as exc:
            raise exc.original_exception
        except Exception as exc:
            self.last_exception = exc
            return (
                None,
                self.last_exception,
                self._assert_failure_timeout(next_interval, exc),
            )

    def _assert_failure_timeout(self, next_interval, exc):
        self.last_exception = exc

        # If next interval is within allowed time period - wait on interval, abort otherwise
        if not self._timeout_exceeded(next_interval):
            if self.logger is not None and self.verbose:
                self.logger.debug(
                    f"Operation not yet successful, Retrying in {next_interval} seconds."
                    f" exc: {mlrun.errors.err_to_str(exc)}"
                )
            return True
        else:
            return False

    def _raise_last_exception(self):
        if self.logger is not None:
            self.logger.warning(
                f"Operation did not complete on time. last exception: {self.last_exception}"
            )

        raise mlrun.errors.MLRunRetryExhaustedError(
            f"Failed to execute command by the given deadline."
            f" last_exception: {self.last_exception},"
            f" function_name: {self.function.__name__},"
            f" timeout: {self.timeout}"
        ) from self.last_exception

    def _timeout_exceeded(self, next_interval=None):
        now = time.monotonic()
        if next_interval:
            now = now + next_interval
        return self.timeout is not None and now >= self.start_time + self.timeout


class AsyncRetryer(Retryer):
    async def run(self):
        self._prepare()
        while not self._timeout_exceeded():
            next_interval = self.first_interval or next(self.backoff)
            result, exc, retry = await self._perform_call(next_interval)
            if retry:
                await asyncio.sleep(next_interval)
            elif not exc:
                return result
            else:
                break

        self._raise_last_exception()

    async def _perform_call(self, next_interval):
        try:
            result = await self.function(*self.args, **self.kwargs)
            return result, None, False
        except mlrun.errors.MLRunFatalFailureError as exc:
            raise exc.original_exception
        except Exception as exc:
            return (
                None,
                self.last_exception,
                self._assert_failure_timeout(next_interval, exc),
            )
