import json
import os
import string
from datetime import datetime, timedelta
from random import randint, choice

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from v3io.dataplane import RaiseForStatus
from v3io_frames import frames_pb2 as fpb2
from v3io_frames.errors import CreateError

from mlrun.api.api.endpoints.model_endpoints import (
    get_endpoint_id,
    _get_endpoint_kv_record_by_id,
    ENDPOINTS_TABLE_PATH,
    ENDPOINT_EVENTS_TABLE_PATH,
)
from mlrun.config import config
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client
from mlrun.api.schemas import (
    ModelEndpointMetadata,
    ModelEndpointSpec,
    ModelEndpoint,
    ObjectStatus,
)


def is_env_params_dont_exist() -> bool:
    return not all(
        (
            os.environ.get(r, False)
            for r in ["V3IO_ACCESS_KEY", "V3IO_API", "V3IO_FRAMESD"]
        )
    )


@pytest.mark.skipif(
    is_env_params_dont_exist(),
    reason="V3IO_ACCESS_KEY, V3IO_API and V3IO_FRAMESD environment must be initialized",
)
def test_clear_endpoint(db: Session, client: TestClient):
    endpoint = _mock_random_endpoint()
    _write_endpoint_to_kv(endpoint)
    endpoint_id = get_endpoint_id(endpoint)

    kv_record = _get_endpoint_kv_record_by_id(endpoint_id)

    assert kv_record

    response = client.post(
        f"/api/projects/{kv_record['project']}/model-endpoints/{endpoint_id}/clear"
    )

    assert response.status_code == 200

    kv_record = _get_endpoint_kv_record_by_id(endpoint_id)

    assert not kv_record


@pytest.mark.skipif(
    is_env_params_dont_exist(),
    reason="V3IO_ACCESS_KEY, V3IO_API and V3IO_FRAMESD environment must be initialized",
)
def test_list_endpoints(db: Session, client: TestClient):
    endpoints_in = [_mock_random_endpoint("active") for _ in range(5)]

    for endpoint in endpoints_in:
        _write_endpoint_to_kv(endpoint)

    response = client.get("/api/projects/test/model-endpoints")

    endpoints_out = [
        ModelEndpoint(**e["endpoint"]) for e in response.json()["endpoints"]
    ]

    endpoints_in_set = {json.dumps(e.dict(), sort_keys=True) for e in endpoints_in}
    endpoints_out_set = {json.dumps(e.dict(), sort_keys=True) for e in endpoints_out}
    endpoints_intersect = endpoints_in_set.intersection(endpoints_out_set)

    assert len(endpoints_intersect) == 5


@pytest.mark.skipif(
    is_env_params_dont_exist(),
    reason="V3IO_ACCESS_KEY, V3IO_API and V3IO_FRAMESD environment must be initialized",
)
def test_list_endpoints_filter(db: Session, client: TestClient):
    for i in range(5):
        endpoint_details = _mock_random_endpoint()

        if i < 1:
            endpoint_details.spec.model = "filterme"

        if i < 2:
            endpoint_details.spec.function = "filterme"

        if i < 3:
            endpoint_details.metadata.tag = "filterme"

        if i < 4:
            endpoint_details.metadata.labels = {"filtermex": "1", "filtermey": "2"}

        _write_endpoint_to_kv(endpoint_details)

    filter_model = json.loads(
        client.get("/api/projects/test/model-endpoints/?model=filterme").text
    )["endpoints"]
    assert len(filter_model) == 1

    filter_function = json.loads(
        client.get("/api/projects/test/model-endpoints/?function=filterme").text
    )["endpoints"]
    assert len(filter_function) == 2

    filter_tag = json.loads(
        client.get("/api/projects/test/model-endpoints/?tag=filterme").text
    )["endpoints"]
    assert len(filter_tag) == 3

    filter_labels = json.loads(
        client.get("/api/projects/test/model-endpoints/?label=filtermex==1").text
    )["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = json.loads(
        client.get(
            "/api/projects/test/model-endpoints/?label=filtermex==1&label=filtermey==2"
        ).text
    )["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = json.loads(
        client.get("/api/projects/test/model-endpoints/?label=filtermey==2").text
    )["endpoints"]
    assert len(filter_labels) == 4


@pytest.mark.skipif(
    is_env_params_dont_exist(),
    reason="V3IO_ACCESS_KEY, V3IO_API and V3IO_FRAMESD environment must be initialized",
)
def test_get_endpoint_metrics(db: Session, client: TestClient):

    start = datetime.utcnow()

    for i in range(5):
        endpoint = _mock_random_endpoint()
        endpoint_id = get_endpoint_id(endpoint)

        _write_endpoint_to_kv(endpoint)

        frames = get_frames_client(container="projects")

        frames.create(
            backend="tsdb", table=ENDPOINT_EVENTS_TABLE_PATH, rate="10/m", if_exists=1
        )

        total = 0

        dfs = []

        for i in range(10):
            count = randint(1, 10)
            total += count
            data = {
                "predictions_per_second_count_1s": count,
                "endpoint_id": endpoint_id,
                "timestamp": start - timedelta(minutes=10 - i),
            }
            df = pd.DataFrame(data=[data])
            dfs.append(df)

        frames.write(
            backend="tsdb",
            table=ENDPOINT_EVENTS_TABLE_PATH,
            dfs=dfs,
            index_cols=["timestamp", "endpoint_id"],
        )

        response = client.get(
            f"/api/projects/test/model-endpoints/{endpoint_id}?metrics=true&name=predictions"
        )
        response = json.loads(response.content)

        assert "metrics" in response

        metrics = response["metrics"]

        assert len(metrics) > 0

        first_metric = metrics[0]

        assert first_metric["name"] == "predictions_per_second"

        response_total = sum((m[1] for m in first_metric["values"]))

        assert total == response_total


def _mock_random_endpoint(state: str = "") -> ModelEndpoint:
    return ModelEndpoint(
        metadata=ModelEndpointMetadata(
            project="test",
            tag=f"v{randint(0, 100)}",
            labels={
                f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)
            },
        ),
        spec=ModelEndpointSpec(
            model=f"model_{randint(0, 100)}",
            function=f"function_{randint(0, 100)}",
            model_class="classifier",
        ),
        status=ObjectStatus(state=state),
    )


def _write_endpoint_to_kv(endpoint: ModelEndpoint):
    endpoint_id = get_endpoint_id(endpoint)

    get_v3io_client().kv.put(
        container=config.model_endpoint_monitoring_container,
        table_path=ENDPOINTS_TABLE_PATH,
        key=endpoint_id,
        attributes={
            "project": endpoint.metadata.project,
            "function": endpoint.spec.function,
            "model": endpoint.spec.model,
            "tag": endpoint.metadata.tag,
            "model_class": endpoint.spec.model_class,
            "labels": json.dumps(endpoint.metadata.labels),
            **{f"_{k}": v for k, v in endpoint.metadata.labels.items()},
        },
    )


@pytest.fixture(autouse=True)
def cleanup_endpoints(db: Session, client: TestClient):
    v3io = get_v3io_client()
    frames = get_frames_client(container="projects")
    try:
        all_records = v3io.kv.new_cursor(
            container=config.model_endpoint_monitoring_container,
            table_path=ENDPOINTS_TABLE_PATH,
            raise_for_status=RaiseForStatus.never,
        ).all()

        all_records = [r["__name"] for r in all_records]

        # Cleanup KV
        for record in all_records:
            get_v3io_client().kv.delete(
                container=config.model_endpoint_monitoring_container,
                table_path=ENDPOINTS_TABLE_PATH,
                key=record,
                raise_for_status=RaiseForStatus.never,
            )
    except RuntimeError:
        pass

    try:
        # Cleanup TSDB
        frames.delete(
            backend="tsdb", table=ENDPOINT_EVENTS_TABLE_PATH, if_missing=fpb2.IGNORE,
        )
    except CreateError:
        pass
