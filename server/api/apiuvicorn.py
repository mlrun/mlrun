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

import uvicorn


def _get_uvicorn_log_config(formatter_class):
    base_log_config = uvicorn.config.LOGGING_CONFIG

    base_log_config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    base_log_config["formatters"]["default"] = {"()": formatter_class}
    return base_log_config


def run(logger, httpdb_config):
    logger.info(
        "Starting API server",
        port=httpdb_config.port,
        debug=httpdb_config.debug,
    )
    uvicorn.run(
        "server.api.main:app",
        host="0.0.0.0",
        port=httpdb_config.port,
        access_log=False,
        timeout_keep_alive=httpdb_config.http_connection_timeout_keep_alive,
        log_config=_get_uvicorn_log_config(
            logger._logger.handlers[0].formatter.__class__
        ),
    )
