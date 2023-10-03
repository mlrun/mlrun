import uvicorn
import uvicorn.logging

import mlrun.utils


# This class is a mixin that combines the default uvicorn formatter with the MLRun once
# This allows us to get a unified log format for all the logs in the system
class UvicornMLRunLoggerMixin(
    uvicorn.logging.DefaultFormatter, mlrun.utils.HumanReadableFormatter
):
    pass


def _get_uvicorn_log_config():
    base_log_config = uvicorn.config.LOGGING_CONFIG
    base_log_config["formatters"]["default"][
        "()"
    ] = "server.api.apiuvicorn.UvicornMLRunLoggerMixin"
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
        log_config=_get_uvicorn_log_config(),
    )
