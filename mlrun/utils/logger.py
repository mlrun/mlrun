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

import json
import logging
from enum import Enum
from sys import stdout
from traceback import format_exception
from typing import IO, Optional, Union

from mlrun.config import config


class JSONFormatter(logging.Formatter):
    def __init__(self):
        super(JSONFormatter, self).__init__()
        self._json_encoder = json.JSONEncoder()

    def format(self, record):
        record_with = getattr(record, "with", {})
        if record.exc_info:
            record_with.update(exc_info=format_exception(*record.exc_info))
        record_fields = {
            "datetime": self.formatTime(record, self.datefmt),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "with": record_with,
        }

        return self._json_encoder.encode(record_fields)


class HumanReadableFormatter(logging.Formatter):
    def format(self, record) -> str:
        record_with = self._record_with(record)
        more = f": {record_with}" if record_with else ""
        return f"> {self.formatTime(record, self.datefmt)} [{record.levelname.lower()}] {record.getMessage()}{more}"

    def _record_with(self, record):
        record_with = getattr(record, "with", {})
        if record.exc_info:
            record_with.update(exc_info=format_exception(*record.exc_info))
        return record_with


class HumanReadableExtendedFormatter(HumanReadableFormatter):
    def format(self, record) -> str:
        record_with = self._record_with(record)
        more = f": {record_with}" if record_with else ""
        return (
            "> "
            f"{self.formatTime(record, self.datefmt)} "
            f"[{record.name}:{record.levelname.lower()}] "
            f"{record.getMessage()}{more}"
        )


class Logger(object):
    def __init__(
        self,
        level,
        name="mlrun",
        propagate=True,
        logger: Optional[logging.Logger] = None,
    ):
        self._logger = logger or logging.getLogger(name)
        self._logger.propagate = propagate
        self._logger.setLevel(level)
        self._bound_variables = {}

        for log_level_func in [
            self.exception,
            self.error,
            self.warn,
            self.warning,
            self.info,
            self.debug,
        ]:
            setattr(self, f"{log_level_func.__name__}_with", log_level_func)

    def set_handler(
        self, handler_name: str, file: IO[str], formatter: logging.Formatter
    ):
        # check if there's a handler by this name
        for handler in self._logger.handlers:
            if handler.name == handler_name:
                self._logger.removeHandler(handler)
                break

        # create a stream handler from the file
        stream_handler = logging.StreamHandler(file)
        stream_handler.name = handler_name

        # set the formatter
        stream_handler.setFormatter(formatter)

        # add the handler to the logger
        self._logger.addHandler(stream_handler)

    def get_child(self, suffix):
        """
        Get a child logger with the given suffix.
        This is useful for when you want to have a logger for a specific component.
        Once the formatter will support logger name, it will be easier to understand
        which component logged the message.

        :param suffix: The suffix to add to the logger name.
        """
        return Logger(
            self.level,
            # name is not set as it is provided by the "getChild"
            name="",
            # allowing child to delegate events logged to ancestor logger
            # not doing so, will leave log lines not being handled
            propagate=True,
            logger=self._logger.getChild(suffix),
        )

    @property
    def level(self):
        return self._logger.level

    def set_logger_level(self, level: Union[str, int]):
        self._logger.setLevel(level)

    def replace_handler_stream(self, handler_name: str, file: IO[str]):
        for handler in self._logger.handlers:
            if handler.name == handler_name:
                handler.stream = file
                return
        raise ValueError(f"Logger does not have a handler named '{handler_name}'")

    def debug(self, message, *args, **kw_args):
        self._update_bound_vars_and_log(logging.DEBUG, message, *args, **kw_args)

    def info(self, message, *args, **kw_args):
        self._update_bound_vars_and_log(logging.INFO, message, *args, **kw_args)

    def warn(self, message, *args, **kw_args):
        self._update_bound_vars_and_log(logging.WARNING, message, *args, **kw_args)

    def warning(self, message, *args, **kw_args):
        self.warn(message, *args, **kw_args)

    def error(self, message, *args, **kw_args):
        self._update_bound_vars_and_log(logging.ERROR, message, *args, **kw_args)

    def exception(self, message, *args, exc_info=True, **kw_args):
        self._update_bound_vars_and_log(
            logging.ERROR, message, *args, exc_info=exc_info, **kw_args
        )

    def bind(self, **kw_args):
        self._bound_variables.update(kw_args)

    def _update_bound_vars_and_log(
        self, level, message, *args, exc_info=None, **kw_args
    ):
        kw_args.update(self._bound_variables)

        if kw_args:
            self._logger.log(
                level, message, *args, exc_info=exc_info, extra={"with": kw_args}
            )
            return

        self._logger.log(level, message, *args, exc_info=exc_info)


class FormatterKinds(Enum):
    HUMAN = "human"
    HUMAN_EXTENDED = "human_extended"
    JSON = "json"


def _create_formatter_instance(formatter_kind: FormatterKinds) -> logging.Formatter:
    return {
        FormatterKinds.HUMAN: HumanReadableFormatter(),
        FormatterKinds.HUMAN_EXTENDED: HumanReadableExtendedFormatter(),
        FormatterKinds.JSON: JSONFormatter(),
    }[formatter_kind]


def create_logger(
    level: str = None,
    formatter_kind: str = FormatterKinds.HUMAN.name,
    name: str = "mlrun",
    stream=stdout,
):
    level = level or config.log_level or "info"

    level = logging.getLevelName(level.upper())

    # create logger instance
    logger_instance = Logger(level, name=name, propagate=False)

    # resolve formatter
    formatter_instance = _create_formatter_instance(
        FormatterKinds(formatter_kind.lower())
    )

    # set handler
    logger_instance.set_handler("default", stream or stdout, formatter_instance)

    return logger_instance
