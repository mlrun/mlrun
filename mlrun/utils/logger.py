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

import sys
import json
import logging
from enum import Enum
from typing import IO


class JSONFormatter(logging.Formatter):
    def __init__(self):
        super(JSONFormatter, self).__init__()
        self._json_encoder = json.JSONEncoder()

    def format(self, record):
        record_fields = {
            'datetime': self.formatTime(record, self.datefmt),
            'level': record.levelname.lower(),
            'message': record.getMessage(),
            'with': getattr(record, 'with', {}),
        }

        return self._json_encoder.encode(record_fields)


class HumanReadableFormatter(logging.Formatter):

    def __init__(self):
        super(HumanReadableFormatter, self).__init__()

    def format(self, record):
        record_with = getattr(record, 'with', {})
        more = f': {record_with}' if record_with else ''
        return f'> {self.formatTime(record, self.datefmt)} [{record.levelname.lower()}] {record.getMessage()}{more}'


class Logger(object):

    def __init__(self, level, name='mlrun', propagate=True):
        self._logger = logging.getLogger(name)
        self._logger.propagate = propagate
        self._logger.setLevel(level)
        self._bound_variables = {}
        self._handlers = {}

    def set_handler(self, handler_name: str, file: IO[str], formatter: logging.Formatter):

        # check if there's a handler by this name
        if handler_name in self._handlers:

            # log that we're removing it
            self.info('Replacing logger output', handler_name=handler_name)

            self._logger.removeHandler(self._handlers[handler_name])

        # create a stream handler from the file
        stream_handler = logging.StreamHandler(file)

        # set the formatter
        stream_handler.setFormatter(formatter)

        # add the handler to the logger
        self._logger.addHandler(stream_handler)

        # save as the named output
        self._handlers[handler_name] = stream_handler

    def debug(self, message, *args, **kw_args):
        self._update_bound_vars_and_log(logging.DEBUG, message, *args, **kw_args)

    def info(self, message, *args, **kw_args):
        self._update_bound_vars_and_log(logging.INFO, message, *args, **kw_args)

    def warn(self, message, *args, **kw_args):
        self._update_bound_vars_and_log(logging.WARNING, message, *args, **kw_args)

    def error(self, message, *args, **kw_args):
        self._update_bound_vars_and_log(logging.ERROR, message, *args, **kw_args)

    def bind(self, **kw_args):
        self._bound_variables.update(kw_args)

    def _update_bound_vars_and_log(self, level, message, *args, **kw_args):
        kw_args.update(self._bound_variables)

        if kw_args:
            self._logger._log(level, message, args, extra={'with': kw_args})
            return

        self._logger._log(level, message, args)


class LoggerFormatterEnum(Enum):
    HUMAN = 'human'
    JSON = 'json'


def _resolve_formatter(logger_formatter: LoggerFormatterEnum):
    return {
        LoggerFormatterEnum.HUMAN: HumanReadableFormatter(),
        LoggerFormatterEnum.JSON: JSONFormatter(),
    }[logger_formatter]


def create_logger(level: str = "debug",
                  formatter: str = LoggerFormatterEnum.HUMAN.name,
                  name: str = "mlrun",
                  stream=sys.stdout):
    level = logging.getLevelName(level.upper())

    # create logger instance
    logger_instance = Logger(level, name=name, propagate=False)

    # resolve formatter
    formatter = _resolve_formatter(LoggerFormatterEnum(formatter.lower()))

    # set handler
    logger_instance.set_handler("default", stream or sys.stdout, formatter)

    # disable event propagation to higher classes
    # TODO: Why?!
    logger_instance.propagate = False

    return logger_instance
