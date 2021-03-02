import json
import os
import string
from datetime import datetime, timedelta
from random import randint, choice
from typing import Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from v3io.dataplane import RaiseForStatus
from v3io_frames import frames_pb2 as fpb2
from v3io_frames.errors import CreateError

from mlrun.api.crud.model_endpoints import (
    get_endpoint_kv_record_by_id,
    ENDPOINT_EVENTS_TABLE_PATH,
    ENDPOINTS_TABLE_PATH,
    ModelEndpoints,
)
from mlrun.api.schemas import ModelEndpoint
from mlrun.config import config
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client

ENV_PARAMS = {"V3IO_ACCESS_KEY", "V3IO_API", "V3IO_FRAMESD"}


def _build_skip_message():
    return f"One of the required environment params is not initialized ({', '.join(ENV_PARAMS)})"


def _is_env_params_dont_exist() -> bool:
    return not all((os.environ.get(r, False) for r in ENV_PARAMS))


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_clear_endpoint(db: Session, client: TestClient):
    access_key = _get_access_key()
    endpoint = _mock_random_endpoint()
    ModelEndpoints.persist_to_kv(access_key, endpoint)

    kv_record = get_endpoint_kv_record_by_id(access_key, endpoint.id)

    assert kv_record
    response = client.post(
        url=f"/api/projects/{kv_record['project']}/model-endpoints/{endpoint.id}/clear",
        headers={"X-V3io-Session-Key": access_key},
    )

    assert response.status_code == 204

    kv_record = get_endpoint_kv_record_by_id(access_key, endpoint.id)

    assert not kv_record


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_update_endpoint(db: Session, client: TestClient):
    access_key = _get_access_key()
    endpoint = _mock_random_endpoint()
    ModelEndpoints.persist_to_kv(access_key, endpoint)

    kv_record_before_update = get_endpoint_kv_record_by_id(access_key, endpoint.id)

    assert kv_record_before_update["state"] == ""

    response = client.post(
        url=f"/api/projects/{endpoint.metadata.project}/model-endpoints/{endpoint.id}/update",
        headers={"X-V3io-Session-Key": access_key},
        json=dict(state="testing...testing...1 2 1 2"),
    )

    assert response.status_code == 204

    kv_record_after_update = get_endpoint_kv_record_by_id(access_key, endpoint.id)

    assert kv_record_after_update["state"] == "testing...testing...1 2 1 2"


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_update_endpoint_doesnt_exists(db: Session, client: TestClient):
    access_key = _get_access_key()
    endpoint = _mock_random_endpoint()

    response = client.post(
        url=f"/api/projects/{endpoint.metadata.project}/model-endpoints/{endpoint.id}/update",
        headers={"X-V3io-Session-Key": access_key},
        json=dict(status="testing...testing...1 2 1 2"),
    )

    assert response.status_code == 400


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_update_endpoint_missing_payload_fields(db: Session, client: TestClient):
    access_key = _get_access_key()
    endpoint = _mock_random_endpoint()
    ModelEndpoints.persist_to_kv(access_key, endpoint)

    kv_record_before_update = get_endpoint_kv_record_by_id(access_key, endpoint.id)

    assert kv_record_before_update

    response = client.post(
        url=f"/api/projects/{endpoint.metadata.project}/model-endpoints/{endpoint.id}/update",
        headers={"X-V3io-Session-Key": access_key},
        json={},
    )

    assert response.status_code == 400


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_list_endpoints(db: Session, client: TestClient):
    endpoints_in = [_mock_random_endpoint("testing") for _ in range(5)]

    for endpoint in endpoints_in:
        ModelEndpoints.persist_to_kv(_get_access_key(), endpoint)

    response = client.get(
        url="/api/projects/test/model-endpoints",
        headers={"X-V3io-Session-Key": _get_access_key()},
    )

    endpoints_out = [
        ModelEndpoint(**e["endpoint"]) for e in response.json()["endpoints"]
    ]

    in_endpoint_ids = set(map(lambda e: e.id, endpoints_in))
    out_endpoint_ids = set(map(lambda e: e.id, endpoints_out))

    endpoints_intersect = in_endpoint_ids.intersection(out_endpoint_ids)
    assert len(endpoints_intersect) == 5


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_list_endpoints_filter(db: Session, client: TestClient):
    access_key = _get_access_key()
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

        ModelEndpoints.persist_to_kv(_get_access_key(), endpoint_details)

    filter_model = json.loads(
        client.get(
            "/api/projects/test/model-endpoints/?model=filterme",
            headers={"X-V3io-Session-Key": access_key},
        ).text
    )["endpoints"]
    assert len(filter_model) == 1

    filter_function = json.loads(
        client.get(
            "/api/projects/test/model-endpoints/?function=filterme",
            headers={"X-V3io-Session-Key": access_key},
        ).text
    )["endpoints"]
    assert len(filter_function) == 2

    filter_tag = json.loads(
        client.get(
            "/api/projects/test/model-endpoints/?tag=filterme",
            headers={"X-V3io-Session-Key": access_key},
        ).text
    )["endpoints"]
    assert len(filter_tag) == 3

    filter_labels = json.loads(
        client.get(
            "/api/projects/test/model-endpoints/?label=filtermex=1",
            headers={"X-V3io-Session-Key": access_key},
        ).text
    )["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = json.loads(
        client.get(
            "/api/projects/test/model-endpoints/?label=filtermex=1&label=filtermey=2",
            headers={"X-V3io-Session-Key": access_key},
        ).text
    )["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = json.loads(
        client.get(
            "/api/projects/test/model-endpoints/?label=filtermey=2",
            headers={"X-V3io-Session-Key": access_key},
        ).text
    )["endpoints"]
    assert len(filter_labels) == 4


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_get_endpoint_metrics(db: Session, client: TestClient):
    frames = get_frames_client(
        token=_get_access_key(), container="projects", address=config.v3io_framesd,
    )

    start = datetime.utcnow()

    for i in range(5):
        endpoint = _mock_random_endpoint()
        ModelEndpoints.persist_to_kv(_get_access_key(), endpoint)

        frames.create(
            backend="tsdb",
            table=f"test/{ENDPOINT_EVENTS_TABLE_PATH}",
            rate="10/m",
            if_exists=1,
        )

        total = 0

        dfs = []

        for i in range(10):
            count = randint(1, 10)
            total += count
            data = {
                "predictions_per_second_count_1s": count,
                "endpoint_id": endpoint.id,
                "timestamp": start - timedelta(minutes=10 - i),
            }
            df = pd.DataFrame(data=[data])
            dfs.append(df)

        frames.write(
            backend="tsdb",
            table=f"test/{ENDPOINT_EVENTS_TABLE_PATH}",
            dfs=dfs,
            index_cols=["timestamp", "endpoint_id"],
        )

        response = client.get(
            url=f"/api/projects/test/model-endpoints/{endpoint.id}?metric=predictions",
            headers={"X-V3io-Session-Key": _get_access_key()},
        )
        response = json.loads(response.content)

        assert "metrics" in response

        metrics = response["metrics"]

        assert len(metrics) > 0

        predictions_per_second = metrics["predictions_per_second"]

        assert predictions_per_second["name"] == "predictions_per_second"

        response_total = sum((m[1] for m in predictions_per_second["values"]))

        assert total == response_total


def _mock_random_endpoint(state: Optional[str] = None) -> ModelEndpoint:
    def random_labels():
        return {f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)}

    return ModelEndpoint.new(
        project="test",
        model=f"model_{randint(0, 100)}",
        function=f"function_{randint(0, 100)}",
        tag=f"v{randint(0, 100)}",
        model_class="classifier",
        labels=random_labels(),
        state=state,
    )


def _write_endpoint_to_kv(endpoint: ModelEndpoint):
    client = get_v3io_client(endpoint=config.v3io_api, access_key=_get_access_key())
    client.kv.put(
        container="projects",
        table_path=f"{endpoint.metadata.project}/{ENDPOINTS_TABLE_PATH}/",
        key=endpoint.id,
        attributes={
            "project": endpoint.metadata.project,
            "function": endpoint.spec.function,
            "model": endpoint.spec.model,
            "tag": endpoint.metadata.tag,
            "model_class": endpoint.spec.model_class,
            "model_artifact": endpoint.metadata.model_artifact,
            "stream_path": endpoint.metadata.stream_path,
            "labels": json.dumps(endpoint.metadata.labels),
            "status": endpoint.status.state,
            **{f"_{k}": v for k, v in endpoint.metadata.labels.items()},
        },
    )


@pytest.fixture(autouse=True)
def cleanup_endpoints(db: Session, client: TestClient):
    v3io = get_v3io_client(endpoint=config.v3io_api, access_key=_get_access_key())

    frames = get_frames_client(
        token=_get_access_key(), container="projects", address=config.v3io_framesd,
    )
    try:
        all_records = v3io.kv.new_cursor(
            container="projects",
            table_path=f"test/{ENDPOINTS_TABLE_PATH}",
            raise_for_status=RaiseForStatus.never,
        ).all()

        all_records = [r["__name"] for r in all_records]

        # Cleanup KV
        for record in all_records:
            v3io.kv.delete(
                container="projects",
                table_path=f"test/{ENDPOINTS_TABLE_PATH}",
                key=record,
                raise_for_status=RaiseForStatus.never,
            )
    except RuntimeError:
        pass

    try:
        # Cleanup TSDB
        frames.delete(
            backend="tsdb",
            table=f"test/{ENDPOINT_EVENTS_TABLE_PATH}",
            if_missing=fpb2.IGNORE,
        )
    except CreateError:
        pass


def _get_access_key() -> Optional[str]:
    return os.environ.get("V3IO_ACCESS_KEY")
