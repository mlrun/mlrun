from dataclasses import asdict
from datetime import datetime
from itertools import zip_longest
from typing import Dict, List, Optional

import pandas as pd
from v3io_frames.frames_pb2 import IGNORE

from mlrun.api.api.endpoints.model_endpoints import get_endpoint
from mlrun.monitoring.alert_log import TSDBAlertLog
from mlrun.monitoring.clients import get_v3io_client, get_frames_client
from mlrun.monitoring.constants import (
    ISO_8601,
    ISO_8601_NO_MILLIES,
    ENDPOINT_EVENT_LOG_TABLE,
    ENDPOINT_STATE_LOG_TABLE,
    DEFAULT_CONTAINER,
)
from mlrun.monitoring.endpoint import EndpointKey, EndpointState
from mlrun.utils import logger


class EventStreamProcessor:
    def __init__(self):
        self.active_endpoints: Dict[str, EndpointState] = {}
        self.endpoint_data: Dict[str, dict] = {}
        self.alert_log = TSDBAlertLog(verbose=True)

        # Create TSDB table
        get_frames_client().create(
            backend="tsdb", table=ENDPOINT_EVENT_LOG_TABLE, if_exists=IGNORE
        )

    def handle_event(self, event):
        endpoint_key = endpoint_key_from_event(event)

        # On first interaction with an endpoint, we create a endpoint state instance and store
        if endpoint_key.hash not in self.active_endpoints:
            self.active_endpoints[endpoint_key.hash] = EndpointState(endpoint_key)

        endpoint_state = self.active_endpoints[endpoint_key.hash]

        # Collect endpoint metadata (feature_cols and train set describe). prediction_col is assumed to be 'prediction'
        if endpoint_key.hash not in self.endpoint_data:
            self.endpoint_data[endpoint_key.hash] = get_endpoint(endpoint_key.hash)

        endpoint_data = self.endpoint_data[endpoint_key.hash]

        # Count individual events
        endpoint_state.event_count += 1

        event_outputs = event["resp"]["outputs"]
        event_inputs = event_outputs["inputs"]
        event_predictions = event_outputs["prediction"]
        event_labels = event_outputs.get("labels", [])

        # Handle individual predictions
        for inputs, prediction, label in zip_longest(
            event_inputs, event_predictions, event_labels
        ):
            self.handle_prediction(
                endpoint_key, endpoint_data, endpoint_state, event, inputs, label
            )

    def handle_prediction(
        self, endpoint_key, endpoint_data, endpoint_state, event, inputs, label
    ):
        timestamp = deserialize_timestamp(event["when"])
        if timestamp is None:
            endpoint_state.alert_count += 1
            message = (
                f"SKIPPING EVENT: Event timestamp ({event['when']}) does not match either "
                f"ISO 8601 ({ISO_8601}), nor ISO 8601 without milliseconds ({ISO_8601_NO_MILLIES})"
            )
            logger.exception(message)
            self.alert_log.critical(endpoint_key, message)
            return

        # Construct prediction dictionary with all available data
        prediction = asdict(endpoint_key)
        prediction["features"] = compute_named_features(endpoint_data, inputs)
        prediction["prediction"] = prediction
        prediction["label"] = label
        prediction["microsec"] = event["microsec"]
        prediction["timestamp"] = timestamp

        endpoint_state.predictions.append(prediction)

        # Update date times related metrics
        if endpoint_state.first_event is None:
            endpoint_state.first_event = timestamp
            endpoint_state.last_flushed = timestamp
        endpoint_state.last_event = timestamp

        # Write state to tables
        write_to_kv(endpoint_state)
        write_to_tsdb(endpoint_state)
        write_to_parquet(endpoint_state)


def endpoint_key_from_event(event):
    project, function_with_tag = event["function_uri"].split("/")

    try:
        function, tag = function_with_tag.split(":")
    except ValueError:
        function, tag = function_with_tag, "latest"

    model = event["model"]
    model_class = event.get("class")

    return EndpointKey(project, function, model, tag, model_class)


def compute_named_features(endpoint_data, inputs):
    # Either get feature names from meta data or generate feature names automatically
    named_features = {}
    if "feature_col" in endpoint_data:
        for i, feature in enumerate(inputs):
            named_features[f"f{endpoint_data['feature_col'][i]}"] = feature
    else:
        for i, feature in enumerate(inputs):
            named_features["features"][f"f{i}"] = feature
    return named_features


def deserialize_timestamp(str_timestamp: str) -> Optional[datetime]:
    try:
        timestamp = datetime.strptime(str_timestamp, ISO_8601)
        return timestamp
    except ValueError:
        try:
            timestamp = datetime.strptime(str_timestamp, ISO_8601_NO_MILLIES)
            return timestamp
        except ValueError:
            return None


def write_to_tsdb(endpoint_state: EndpointState):
    dfs = (
        prediction_dict_to_tsdb_readable(prediction)
        for prediction in endpoint_state.predictions
    )
    no_none_dfs = (df for df in dfs if df is not None)
    get_frames_client().write(
        backend="tsdb", table=ENDPOINT_EVENT_LOG_TABLE, dfs=no_none_dfs
    )


def write_to_kv(endpoint_state: EndpointState):
    attributes = asdict(endpoint_state.endpoint_key)

    attributes = {
        **attributes,
        "first_request": endpoint_state.first_event.strftime(ISO_8601),
        "last_request": endpoint_state.last_event.strftime(ISO_8601),
        "requests_per_second": 0,
        "predictions_per_second": 0,
        "uptime": compute_human_readable_uptime(endpoint_state.first_event),
        "average_microsec": compute_average_microsec(endpoint_state.predictions),
        "alert_count": endpoint_state.alert_count,
    }

    get_v3io_client().kv.put(
        container=DEFAULT_CONTAINER,
        table_path=ENDPOINT_STATE_LOG_TABLE,
        key=endpoint_state.endpoint_key.hash,
        attributes=attributes,
    )


def write_to_parquet(endpoint_state: EndpointState):
    pass


def prediction_dict_to_tsdb_readable(prediction: dict) -> Optional[pd.DataFrame]:
    df = pd.DataFrame([prediction])

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format=ISO_8601)
    except ValueError:
        try:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], format=ISO_8601_NO_MILLIES
            )
        except ValueError:
            return None

    df.set_index(["timestamp", "model_hash"], inplace=True)


def compute_human_readable_uptime(datetime_obj: datetime) -> str:
    uptime_seconds = (datetime.now() - datetime_obj).total_seconds()
    days, remainder = divmod(uptime_seconds, 86400)  # seconds in day
    hours, remainder = divmod(remainder, 3600)  # seconds in hear
    minutes, seconds = divmod(remainder, 60)  # seconds in minute

    if days:
        uptime = f"{int(days)}d"
    elif hours:
        uptime = f"{int(hours)}h"
    elif minutes:
        uptime = f"{int(minutes)}m"
    elif seconds:
        uptime = f"{int(seconds)}s"
    else:
        uptime = "N/A"

    return uptime


def compute_average_microsec(predictions: List[dict]) -> float:
    return sum(p["microsec"] for p in predictions) / len(predictions)
