import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from mlrun.monitoring.clients import get_v3io_client, get_frames_client
from mlrun.monitoring.constants import (
    ISO_8601,
    ENDPOINT_ALERT_LOG_TABLE,
    DEFAULT_CONTAINER,
    ENDPOINT_ALERT_LOG_STREAM,
)
from mlrun.monitoring.endpoint import EndpointKey
from mlrun.utils import logger


class AlertLog(ABC):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.alert_count = 0

    @abstractmethod
    def _log(
        self,
        severity: str,
        endpoint_key: EndpointKey,
        message: str,
        meta_data: Optional[dict] = None,
    ):
        pass

    def minor(
        self, endpoint_key: EndpointKey, message: str, meta_data: Optional[dict] = None
    ):
        self.alert_count += 1
        self._log("MINOR", endpoint_key, message, meta_data)

    def major(
        self, endpoint_key: EndpointKey, message: str, meta_data: Optional[dict] = None
    ):
        self.alert_count += 1
        self._log("MAJOR", endpoint_key, message, meta_data)

    def critical(
        self, endpoint_key: EndpointKey, message: str, meta_data: Optional[dict] = None
    ):
        self.alert_count += 1
        self._log("CRITICAL", endpoint_key, message, meta_data)


class StreamAlertLog(AlertLog):
    def __init__(
        self,
        container: str = DEFAULT_CONTAINER,
        stream_path: str = ENDPOINT_ALERT_LOG_STREAM,
        shard_count: int = 1,
        verbose: bool = False,
    ):
        super().__init__(verbose)
        self.container = container
        self.stream_path = stream_path

        get_v3io_client().stream.create(
            container=self.container,
            stream_path=self.stream_path,
            shard_count=shard_count,
        )

    def _log(
        self,
        severity: str,
        endpoint_key: EndpointKey,
        message: str,
        meta_data: Optional[dict] = None,
    ):
        msg = f"[{datetime.now()}] [{severity}] {message}"

        if self.verbose:
            logger.info(msg)

        get_v3io_client().stream.put_records(
            container=self.container,
            stream_path=self.stream_path,
            records=[{"data": msg}],
        )


class TSDBAlertLog(AlertLog):
    def __init__(
        self,
        table: str = ENDPOINT_ALERT_LOG_TABLE,
        verbose: bool = False,
    ):
        super().__init__(verbose)
        self.table = table
        try:
            get_frames_client().create(
                backend="tsdb", table=self.table, rate="1/h", if_exists=1
            )
        except Exception as e:
            logger.exception(e)

    def _log(
        self,
        severity: str,
        endpoint_key: EndpointKey,
        message: str,
        meta_data: Optional[dict] = None,
    ):
        now = str(datetime.now())
        df = pd.DataFrame(
            [
                {
                    "timestamp": now,
                    "ts": now,
                    "sevirity": severity,
                    "message": message,
                    "endpoint_key": endpoint_key.hash,
                    "meta_data": json.dumps(meta_data) if meta_data else "",
                }
            ]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], format=ISO_8601)
        df.set_index(["timestamp", "model_hash"], inplace=True)

        if self.verbose:
            logger.info(message)

        get_frames_client().write(backend="tsdb", table=self.table, dfs=df)
