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

import unittest
from contextlib import nullcontext as does_not_raise

import pytest

from mlrun.utils.http import HTTPSessionWithRetry


@pytest.fixture
def http_session():
    with HTTPSessionWithRetry() as session:
        yield session


def raise_exception():
    try:
        raise ConnectionError("This is an ErrorA")
    except ConnectionError as e1:
        try:
            raise Exception from e1
        except Exception as e2:
            return e2


@pytest.mark.parametrize(
    "error_to_raise,expected",
    [
        # Test ConnectionError and ConnectionRefusedError cases that occur once,
        # and are retryable errors, so we expect no Exception to be raised
        ([ConnectionError("This is an ConnectionErr"), True], does_not_raise()),
        (
            [ConnectionRefusedError("This is a ConnectionRefusedErr"), True],
            does_not_raise(),
        ),
        # Test a custom exception with a root cause that is included in our retryable exceptions list,
        # should not raise an exception
        ([raise_exception(), True], does_not_raise()),
        # Test a custom exception with a root cause that is included in our retryable exceptions list,
        # it will be raised 3 times before we expect an error to be raised
        (raise_exception(), pytest.raises(Exception)),
        # Test a non-retryable error and ensure it fails immediately and is not retried
        ([TypeError("TypeErr"), True], pytest.raises(TypeError)),
    ],
)
def test_session_retry(http_session: HTTPSessionWithRetry, error_to_raise, expected):
    with unittest.mock.patch(
        "mlrun.utils.http.requests.Session.request", side_effect=error_to_raise
    ):
        with expected:
            http_session.request("GET", "http://localhost:30678")
