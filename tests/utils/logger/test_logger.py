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
from io import StringIO
from typing import Generator

import pytest

from mlrun.utils.logger import FormatterKinds, Logger, create_logger


@pytest.fixture(params=[formatter_kind.name for formatter_kind in list(FormatterKinds)])
def make_stream_logger(request) -> (StringIO, Logger):
    stream = StringIO()
    logger = create_logger("debug", request.param, "test-logger", stream)
    yield stream, logger


@pytest.fixture(params=["debug", "warn", "warning", "info", "error", "exception"])
def logger_level(request) -> Generator:
    yield request.param


def test_regular(make_stream_logger):
    stream, test_logger = make_stream_logger
    test_logger.debug("SomeText")
    assert "SomeText" in stream.getvalue()


def test_log_level(make_stream_logger):
    stream, test_logger = make_stream_logger
    test_logger.set_logger_level("INFO")
    test_logger.debug("SomeText")
    assert "SomeText" not in stream.getvalue()


def test_with_args(make_stream_logger):
    stream, test_logger = make_stream_logger
    test_logger.debug("special_arg %s", "special_arg_value")
    assert "special_arg" in stream.getvalue()
    assert "special_arg_value" in stream.getvalue()


def test_with_args_and_kwargs(make_stream_logger):
    stream, test_logger = make_stream_logger
    test_logger.debug(
        "special_arg %s", "special_arg_value", special_kwarg_name="special_kwarg_value"
    )
    assert "special_arg" in stream.getvalue()
    assert "special_arg_value" in stream.getvalue()
    assert "special_kwarg_name" in stream.getvalue()
    assert "special_kwarg_value" in stream.getvalue()


def test_with_kwargs(make_stream_logger):
    stream, test_logger = make_stream_logger
    test_logger.debug("special_arg %s", special_kwarg_name="special_kwarg_value")
    assert "special_arg %s" in stream.getvalue()
    assert "special_kwarg_name" in stream.getvalue()
    assert "special_kwarg_value" in stream.getvalue()


def test_levels(make_stream_logger, logger_level):
    stream, test_logger = make_stream_logger
    getattr(test_logger, logger_level)(
        "Message %s", "somearg", somekwarg="somekwarg-value"
    )
    assert "Message somearg" in stream.getvalue()
    assert "somekwarg" in stream.getvalue()
    assert "somekwarg-value" in stream.getvalue()


def test_exception_with_stack(make_stream_logger):
    stream, test_logger = make_stream_logger
    err = None
    try:
        1 / 0
    except Exception as exc:
        err = exc
        test_logger.exception("This is just a test")
    assert str(err) in stream.getvalue()
    assert "This is just a test" in stream.getvalue()
