import json
import os
import string
from datetime import datetime, timedelta
from random import randint, choice

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from v3io_frames import frames_pb2 as fpb2

from mlrun.api import schemas
from mlrun.api.api.endpoints.model_endpoints import (
    get_endpoint_id,
    _get_endpoint_kv_record_by_id,
    ENDPOINTS,
    ENDPOINT_EVENTS,
)
from mlrun.config import config
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client


def is_env_params_exist() -> bool:
    return not all(
        (
            os.environ.get(r, False)
            for r in ["V3IO_ACCESS_KEY", "V3IO_API", "V3IO_FRAMESD"]
        )
    )


@pytest.mark.skipif(
    is_env_params_exist(),
    reason="Either V3IO_ACCESS_KEY or V3IO_API environment params not found",
)
def test_create_endpoint(db: Session, client: TestClient):
    endpoint = create_random_endpoint()

    response = client.post(
        f"/api/projects/{endpoint.metadata.project}/model-endpoints",
        json=endpoint.dict(),
    )

    assert response.status_code == 200

    endpoint_id = get_endpoint_id(endpoint)

    kv_record = _get_endpoint_kv_record_by_id(
        endpoint_id, ["project", "model", "function", "tag", "labels", "model_class"]
    )

    kv_record.update({"labels": json.loads(kv_record["labels"])})

    assert endpoint.metadata.project == kv_record["project"]
    assert endpoint.spec.model == kv_record["model"]
    assert endpoint.spec.function == kv_record["function"]
    assert endpoint.metadata.tag == kv_record["tag"]
    assert json.dumps(endpoint.metadata.labels, sort_keys=True) == json.dumps(
        kv_record["labels"], sort_keys=True
    )


@pytest.mark.skipif(
    is_env_params_exist(),
    reason="Either V3IO_ACCESS_KEY or V3IO_API environment params not found",
)
def test_delete_endpoint(db: Session, client: TestClient):
    endpoint = create_random_endpoint()

    response = client.post(
        f"/api/projects/{endpoint.metadata.project}/model-endpoints",
        json=endpoint.dict(),
    )

    assert response.status_code == 200

    endpoint_id = get_endpoint_id(endpoint)

    kv_record = _get_endpoint_kv_record_by_id(endpoint_id)

    assert kv_record

    response = client.delete(
        f"/api/projects/{kv_record['project']}/model-endpoints/{endpoint_id}"
    )

    assert response.status_code == 200

    kv_record = _get_endpoint_kv_record_by_id(endpoint_id)

    assert not kv_record


@pytest.mark.skipif(
    is_env_params_exist(),
    reason="Either V3IO_ACCESS_KEY or V3IO_API environment params not found",
)
def test_list_endpoints(db: Session, client: TestClient):
    endpoints_in = [create_random_endpoint("created") for _ in range(5)]

    for endpoint in endpoints_in:
        response = client.post(
            f"/api/projects/test/model-endpoints",
            json=endpoint.dict(),
        )
        assert response.status_code == 200

    response = client.get("/api/projects/test/model-endpoints")

    endpoints_out = [
        schemas.Endpoint(**e["endpoint"])
        for e in json.loads(response.text)["endpoints"]
    ]

    endpoints_in_set = {json.dumps(e.dict(), sort_keys=True) for e in endpoints_in}
    endpoints_out_set = {json.dumps(e.dict(), sort_keys=True) for e in endpoints_out}
    endpoints_intersect = endpoints_in_set.intersection(endpoints_out_set)

    assert len(endpoints_intersect) == 5


@pytest.mark.skipif(
    is_env_params_exist(),
    reason="Either V3IO_ACCESS_KEY or V3IO_API environment params not found",
)
def test_list_endpoints_filter(db: Session, client: TestClient):
    for i in range(5):
        endpoint_details = create_random_endpoint()

        if i < 1:
            endpoint_details.spec.model = "filterme"

        if i < 2:
            endpoint_details.spec.function = "filterme"

        if i < 3:
            endpoint_details.metadata.tag = "filterme"

        if i < 4:
            endpoint_details.metadata.labels = {"filtermex": "1", "filtermey": "2"}

        response = client.post(
            f"/api/projects/test/model-endpoints",
            json=endpoint_details.dict(),
        )

        assert response.status_code == 200

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


@pytest.fixture(autouse=True)
def cleanup_endpoints(db: Session, client: TestClient):
    v3io = get_v3io_client()
    frames = get_frames_client(container="projects")

    all_records = v3io.kv.new_cursor(
        container=config.model_endpoint_monitoring_container, table_path=ENDPOINTS
    ).all()
    all_records = [r["__name"] for r in all_records]

    # Cleanup KV
    for record in all_records:
        get_v3io_client().kv.delete(
            container=config.model_endpoint_monitoring_container,
            table_path=ENDPOINTS,
            key=record,
        )

    # Cleanup TSDB
    frames.delete(
        backend="tsdb",
        table=ENDPOINT_EVENTS,
        if_missing=fpb2.IGNORE,
    )


@pytest.mark.skipif(
    is_env_params_exist(),
    reason="Either V3IO_ACCESS_KEY or V3IO_API environment params not found",
)
def test_get_endpoint_metrics(db: Session, client: TestClient):
    endpoint_id = "test.0dc1b5d623bf5d6584ee5d5ead27a7b2"

    frames = get_frames_client(container="projects")
    v3io = get_v3io_client()

    v3io.kv.put(
        container="projects",
        table_path=ENDPOINTS,
        key=endpoint_id,
        attributes={"test": True},
    )

    frames.create(backend="tsdb", table=ENDPOINT_EVENTS, rate="10/m")

    start = datetime.utcnow()

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
        table=ENDPOINT_EVENTS,
        dfs=dfs,
        index_cols=["timestamp", "endpoint_id"],
    )

    response = client.get(
        f"/api/projects/test/model-endpoints/{endpoint_id}/metrics?name=predictions"
    )

    metric = json.loads(response.content)["metrics"][0]

    assert metric["name"] == "predictions_per_second"

    response_total = sum((m[1] for m in metric["values"]))

    assert total == response_total

    pass


def create_random_endpoint(state: str = "") -> schemas.Endpoint:
    return schemas.Endpoint(
        metadata=schemas.ObjectMetadata(
            name="",
            project="test",
            tag=f"v{randint(0,100)}",
            labels={
                f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)
            },
            updated=None,
            uid=None,
        ),
        spec=schemas.EndpointSpec(
            model=f"model_{randint(0,100)}",
            function=f"function_{randint(0,100)}",
            model_class="classifier",
        ),
        status=schemas.ObjectStatus(state=state),
    )
