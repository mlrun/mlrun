from typing import Generator
from io import StringIO

import pytest

from mlrun.clients.logger import create_logger, LoggerFormatterEnum, Logger


@pytest.fixture(scope="session")
def stream() -> Generator:
    io_stream = StringIO()
    try:
        yield io_stream
    finally:
        io_stream.close()


@pytest.fixture(params=list(LoggerFormatterEnum.__members__))
def test_logger(request, stream) -> Generator:
    yield create_logger("debug", request.param, "test-logger", stream)


def test_regular(test_logger: Logger, stream: StringIO):
    test_logger.debug("SomeText")
    assert "SomeText" in stream.getvalue()


def test_with_args(test_logger: Logger, stream: StringIO):
    test_logger.debug("special_arg %s", "special_arg_value")
    assert "special_arg" in stream.getvalue()
    assert "special_arg_value" in stream.getvalue()


def test_with_args_and_kwargs(test_logger: Logger, stream: StringIO):
    test_logger.debug("special_arg %s", "special_arg_value", name="special_kwarg")
    assert "special_arg" in stream.getvalue()
    assert "special_arg_value" in stream.getvalue()
    assert "special_kwarg" in stream.getvalue()


def test_with_kwargs(test_logger: Logger, stream: StringIO):
    test_logger.debug("special_arg %s", name="special_kwarg")
    assert "special_arg %s" in stream.getvalue()
    assert "special_kwarg" in stream.getvalue()
