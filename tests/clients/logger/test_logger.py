from typing import Generator
from io import StringIO

import pytest

from mlrun.utils.logger import create_logger, LoggerFormatterEnum, Logger


@pytest.fixture(params=list(LoggerFormatterEnum.__members__))
def test_stream_logger(request) -> Generator:
    stream = StringIO()
    logger = create_logger("debug", request.param, "test-logger", stream)
    yield stream, logger


def test_regular(test_stream_logger):
    stream, test_logger = test_stream_logger
    test_logger.debug("SomeText")
    assert "SomeText" in stream.getvalue()


def test_with_args(test_stream_logger):
    stream, test_logger = test_stream_logger
    test_logger.debug("special_arg %s", "special_arg_value")
    assert "special_arg" in stream.getvalue()
    assert "special_arg_value" in stream.getvalue()


def test_with_args_and_kwargs(test_stream_logger):
    stream, test_logger = test_stream_logger
    test_logger.debug("special_arg %s", "special_arg_value", special_kwarg_name="special_kwarg_value")
    assert "special_arg" in stream.getvalue()
    assert "special_arg_value" in stream.getvalue()
    assert "special_kwarg_name" in stream.getvalue()
    assert "special_kwarg_value" in stream.getvalue()


def test_with_kwargs(test_stream_logger):
    stream, test_logger = test_stream_logger
    test_logger.debug("special_arg %s", special_kwarg_name="special_kwarg_value")
    assert "special_arg %s" in stream.getvalue()
    assert "special_kwarg_name" in stream.getvalue()
    assert "special_kwarg_value" in stream.getvalue()
