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
import contextvars
import datetime
import logging
import os
import string
import sys
import typing
from enum import Enum
from functools import cached_property
from sys import stdout
from traceback import format_exception
from typing import IO, Optional, Union

import orjson
import pydantic.v1

from mlrun import errors
from mlrun.config import config

context_id_var = contextvars.ContextVar("context_id", default=None)


class _BaseFormatter(logging.Formatter):
    def _json_dump(self, json_object):
        def default(obj):
            if isinstance(obj, pydantic.v1.BaseModel):
                return obj.dict()

            # EAFP all the way.
            # Leave the unused "exc" in for debugging ease
            try:
                return obj.__log__()
            except Exception as exc:  # noqa
                try:
                    return obj.__repr__()
                except Exception as exc:  # noqa
                    try:
                        return str(obj)
                    except Exception as exc:
                        raise TypeError from exc

        return orjson.dumps(
            json_object,
            option=orjson.OPT_NAIVE_UTC
            | orjson.OPT_SERIALIZE_NUMPY
            | orjson.OPT_NON_STR_KEYS
            | orjson.OPT_SORT_KEYS,
            default=default,
        ).decode()

    def _record_with(self, record):
        record_with = getattr(record, "with", {})
        if record.exc_info:
            record_with.update(exc_info=format_exception(*record.exc_info))
        if "ctx" not in record_with:
            if (ctx_id := context_id_var.get()) is not None:
                record_with["ctx"] = ctx_id
        return record_with


class JSONFormatter(_BaseFormatter):
    def format(self, record) -> str:
        record_with = self._record_with(record)
        record_fields = {
            "datetime": self.formatTime(record, self.datefmt),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "with": record_with,
        }

        return self._json_dump(record_fields)


class HumanReadableFormatter(_BaseFormatter):
    def format(self, record) -> str:
        more = self._resolve_more(record)
        return (
            f"> {self.formatTime(record, self.datefmt)} "
            f"[{record.levelname.lower()}] "
            f"{record.getMessage().rstrip()}"
            f"{more}"
        )

    def _resolve_more(self, record):
        record_with = self._record_with(record)
        record_with_encoded = self._json_dump(record_with) if record_with else ""
        more = f": {record_with_encoded}" if record_with_encoded else ""
        return more


class CustomFormatter(HumanReadableFormatter):
    """
    To enable custom logger formatter, configure MLRun with the following env variables:
    1. "MLRUN_LOG_FORMATTER" = "custom" - change the default log formatter.
    2. "MLRUN_LOG_FORMAT_OVERRIDE" = "> {timestamp} [{level}] Running module: {module} {message} {more}" - logger format
        * Note that your custom format must include those 4 fields - timestamp, level, message and more
    If the custom format is not configured properly , MLRun will use the default logger (human format).
    """

    # This attribute is used to solve an issue
    # that causes the warning to be written numerous times(for any log generation).
    # We want to print the errors just once, not for each logger generation.
    fail_on_format_configuration = False  # for issues that relates to unrecognized keys
    fail_on_missing_default_keys_key = (
        False  # for issues that relates to missing default keys
    )

    def format(self, record) -> str:
        more = self._resolve_more(record)
        custom_format = config.log_format_override
        _custom_format = None
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        try:
            if custom_format:
                default_keys = ["timestamp", "level", "message", "more"]
                formatter = string.Formatter()
                custom_format_keys = [
                    key
                    for _, key, _, _ in formatter.parse(custom_format)
                    if key is not None
                ]
                missing_default_flags = list(
                    set(default_keys) - set(custom_format_keys)
                )

                if (
                    missing_default_flags
                    and not CustomFormatter.fail_on_missing_default_keys_key
                ):
                    print(
                        f'> {formatted_time} [warning] Custom loggers must '
                        f'include those keys within the logger format, {", ".join(default_keys)} '
                        f'your format is missing: {", ".join(missing_default_flags)}',
                        file=sys.stderr,
                    )
                    CustomFormatter.fail_on_missing_default_keys_key = True
                record_dict = record.__dict__
                missing_format_configuraiton_keys = list(
                    set(custom_format_keys)
                    - set(default_keys)
                    - set(record_dict.keys())
                )
                if missing_format_configuraiton_keys:
                    if not CustomFormatter.fail_on_format_configuration:
                        print(
                            f"> {formatted_time} [warning] Failed to create custom logger due "
                            f'to missing format key in the log record: {", ".join(missing_format_configuraiton_keys)}',
                            file=sys.stderr,
                        )
                        CustomFormatter.fail_on_format_configuration = True
                    _format = (
                        f"> {self.formatTime(record, self.datefmt)} "
                        f"[{record.levelname.lower()}] "
                        f"{record.getMessage().rstrip()}"
                        f"{more}"
                    )
                _custom_format = custom_format.format(
                    timestamp=self.formatTime(record, self.datefmt),
                    level=record.levelname.lower(),
                    message=record.getMessage().rstrip(),
                    more=more or "",
                    **record_dict,
                )
                CustomFormatter.fail_on_format_configuration = True
        except Exception as e:
            if not CustomFormatter.fail_on_format_configuration:
                print(
                    f"> {formatted_time} [warning] Failed to create custom logger, "
                    f"see Exception: {errors.err_to_str(e)}",
                    file=sys.stderr,
                )
                CustomFormatter.fail_on_format_configuration = True
        _format = _custom_format or (
            f"> {self.formatTime(record, self.datefmt)} "
            f"[{record.levelname.lower()}] "
            f"{record.getMessage().rstrip()}"
            f"{more}"
        )
        return _format


class HumanReadableExtendedFormatter(HumanReadableFormatter):
    _colors = {
        logging.NOTSET: "",
        logging.DEBUG: "\x1b[34m",
        logging.INFO: "\x1b[36m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[0;31m",
        logging.CRITICAL: "\x1b[1;31m",
    }
    _color_reset = "\x1b[0m"

    def format(self, record) -> str:
        more = ""
        record_with = self._record_with(record)
        if record_with:

            def _format_value(val):
                formatted_val = (
                    val
                    if isinstance(val, str)
                    else str(orjson.loads(self._json_dump(val)))
                )
                return (
                    formatted_val.replace("\n", "\n\t\t")
                    if len(formatted_val) < 4096
                    else repr(formatted_val)
                )

            more = "\n\t" + "\n\t".join(
                [f"{key}: {_format_value(val)}" for key, val in record_with.items()]
            )
        return (
            f"{self._get_message_color(record.levelno)}> "
            f"{self.formatTime(record, self.datefmt)} "
            f"[{record.name}:{record.levelname.lower()}] "
            f"{record.getMessage()}{more}{self._get_color_reset()}"
        )

    def _get_color_reset(self):
        if not self._have_color_support:
            return ""

        return self._color_reset

    def _get_message_color(self, levelno):
        if not self._have_color_support:
            return ""

        return self._colors[levelno]

    @cached_property
    def _have_color_support(self):
        if os.environ.get("PYCHARM_HOSTED"):
            return True
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("CLICOLOR_FORCE"):
            return True
        return stdout.isatty()


class Logger:
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

    def set_logger_level(self, level: Union[str, int]) -> None:
        self._logger.setLevel(level)

    def replace_handler_stream(self, handler_name: str, file: IO[str]) -> None:
        self.get_handler(handler_name).stream = file

    def get_handler(self, name: str) -> logging.Handler:
        for handler in self._logger.handlers:
            if handler.name == name:
                return handler
        raise ValueError(f"Logger does not have a handler named '{name}'")

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
    CUSTOM = "custom"


def resolve_formatter_by_kind(
    formatter_kind: FormatterKinds,
) -> type[
    typing.Union[
        HumanReadableFormatter,
        HumanReadableExtendedFormatter,
        JSONFormatter,
        CustomFormatter,
    ]
]:
    return {
        FormatterKinds.HUMAN: HumanReadableFormatter,
        FormatterKinds.HUMAN_EXTENDED: HumanReadableExtendedFormatter,
        FormatterKinds.JSON: JSONFormatter,
        FormatterKinds.CUSTOM: CustomFormatter,
    }[formatter_kind]


def create_test_logger(name: str = "mlrun", stream: IO[str] = stdout) -> Logger:
    return create_logger(
        level="debug",
        formatter_kind=FormatterKinds.HUMAN_EXTENDED.name,
        name=name,
        stream=stream,
    )


def create_logger(
    level: Optional[str] = None,
    formatter_kind: str = FormatterKinds.HUMAN.name,
    name: str = "mlrun",
    stream: IO[str] = stdout,
) -> Logger:
    level = level or config.log_level or "info"

    level = logging.getLevelName(level.upper())

    # create logger instance
    logger_instance = Logger(level, name=name, propagate=False)

    # resolve formatter
    formatter_instance = resolve_formatter_by_kind(
        FormatterKinds(formatter_kind.lower())
    )

    # set handler
    logger_instance.set_handler("default", stream or stdout, formatter_instance())

    return logger_instance
