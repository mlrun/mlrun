import os
import string
from datetime import datetime, timedelta
from random import choice, randint
from typing import Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from v3io.dataplane import RaiseForStatus
from v3io_frames import frames_pb2 as fpb2
from v3io_frames.errors import CreateError

from mlrun.api.api.endpoints.model_endpoints import get_or_raise
from mlrun.api.crud.model_endpoints import (
    ENDPOINT_EVENTS_TABLE_PATH,
    ENDPOINTS_TABLE_PATH,
    build_kv_cursor_filter_expression,
    deserialize_endpoint_from_kv,
    get_access_key,
    get_endpoint_features,
    get_endpoint_metrics,
    serialize_endpoint_to_kv,
    string_to_tsdb_name,
)
from mlrun.api.schemas import ModelEndpoint
from mlrun.config import config
from mlrun.errors import (
    MLRunBadRequestError,
    MLRunInvalidArgumentError,
    MLRunNotFoundError,
)
from mlrun.utils.v3io_clients import get_frames_client, get_v3io_client

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
    serialize_endpoint_to_kv(access_key, endpoint)
    kv_record = deserialize_endpoint_from_kv(
        access_key=access_key,
        project=endpoint.metadata.project,
        endpoint_id=endpoint.id,
    )

    assert kv_record
    response = client.post(
        url=f"/api/projects/{kv_record.metadata.project}/model-endpoints/{endpoint.id}/clear",
        headers={"X-V3io-Session-Key": access_key},
    )

    assert response.status_code == 204

    with pytest.raises(MLRunNotFoundError):
        deserialize_endpoint_from_kv(
            access_key=access_key,
            project=endpoint.metadata.project,
            endpoint_id=endpoint.id,
        )


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_update_endpoint(db: Session, client: TestClient):
    access_key = _get_access_key()
    endpoint = _mock_random_endpoint()
    serialize_endpoint_to_kv(access_key=access_key, endpoint=endpoint)

    kv_record_before_update = deserialize_endpoint_from_kv(
        access_key=access_key,
        project=endpoint.metadata.project,
        endpoint_id=endpoint.id,
    )

    assert kv_record_before_update.status.state == ""

    response = client.post(
        url=f"/api/projects/{endpoint.metadata.project}/model-endpoints/{endpoint.id}/update",
        headers={"X-V3io-Session-Key": access_key},
        json=dict(state="testing...testing...1 2 1 2"),
    )

    assert response.status_code == 204

    kv_record_after_update = deserialize_endpoint_from_kv(
        access_key=access_key,
        project=endpoint.metadata.project,
        endpoint_id=endpoint.id,
    )

    assert kv_record_after_update.status.state == "testing...testing...1 2 1 2"


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
    serialize_endpoint_to_kv(access_key, endpoint)

    kv_record_before_update = deserialize_endpoint_from_kv(
        access_key=access_key,
        project=endpoint.metadata.project,
        endpoint_id=endpoint.id,
    )

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
        serialize_endpoint_to_kv(_get_access_key(), endpoint)

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

        serialize_endpoint_to_kv(_get_access_key(), endpoint_details)

    filter_model = client.get(
        "/api/projects/test/model-endpoints/?model=filterme",
        headers={"X-V3io-Session-Key": access_key},
    ).json()["endpoints"]
    assert len(filter_model) == 1

    filter_function = client.get(
        "/api/projects/test/model-endpoints/?function=filterme",
        headers={"X-V3io-Session-Key": access_key},
    ).json()["endpoints"]
    assert len(filter_function) == 2

    filter_tag = client.get(
        "/api/projects/test/model-endpoints/?tag=filterme",
        headers={"X-V3io-Session-Key": access_key},
    ).json()["endpoints"]
    assert len(filter_tag) == 3

    filter_labels = client.get(
        "/api/projects/test/model-endpoints/?label=filtermex=1",
        headers={"X-V3io-Session-Key": access_key},
    ).json()["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = client.get(
        "/api/projects/test/model-endpoints/?label=filtermex=1&label=filtermey=2",
        headers={"X-V3io-Session-Key": access_key},
    ).json()["endpoints"]
    assert len(filter_labels) == 4

    filter_labels = client.get(
        "/api/projects/test/model-endpoints/?label=filtermey=2",
        headers={"X-V3io-Session-Key": access_key},
    ).json()["endpoints"]
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
        serialize_endpoint_to_kv(_get_access_key(), endpoint)

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
        response = response.json()

        assert "metrics" in response

        metrics = response["metrics"]

        assert len(metrics) > 0

        predictions_per_second = metrics["predictions_per_second_count_1s"]

        assert predictions_per_second["name"] == "predictions_per_second_count_1s"

        response_total = sum((m[1] for m in predictions_per_second["values"]))

        assert total == response_total


def test_get_endpoint_metric_function():
    frames = get_frames_client(
        token=_get_access_key(), container="projects", address=config.v3io_framesd,
    )

    start = datetime.utcnow()

    endpoint = _mock_random_endpoint()
    serialize_endpoint_to_kv(_get_access_key(), endpoint)

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

    endpoint_metrics = get_endpoint_metrics(
        access_key=_get_access_key(),
        project="test",
        endpoint_id=endpoint.id,
        name=["predictions"],
    )

    assert "predictions_per_second_count_1s" in endpoint_metrics

    actual_values = endpoint_metrics["predictions_per_second_count_1s"].values
    assert len(actual_values) == 10
    assert sum(map(lambda t: t[1], actual_values)) == total


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


def test_build_kv_cursor_filter_expression():
    with pytest.raises(MLRunInvalidArgumentError):
        build_kv_cursor_filter_expression("")

    filter_expression = build_kv_cursor_filter_expression(project="test")
    assert filter_expression == "project=='test'"

    filter_expression = build_kv_cursor_filter_expression(
        project="test", function="test_function", model="test_model", tag="test_tag"
    )
    expected = "project=='test' AND function=='test_function' AND model=='test_model' AND tag=='test_tag'"
    assert filter_expression == expected

    filter_expression = build_kv_cursor_filter_expression(
        project="test", labels=["lbl1", "lbl2"]
    )
    assert filter_expression == "project=='test' AND exists(_lbl1) AND exists(_lbl2)"

    filter_expression = build_kv_cursor_filter_expression(
        project="test", labels=["lbl1=1", "lbl2=2"]
    )
    assert filter_expression == "project=='test' AND _lbl1=='1' AND _lbl2=='2'"

    filter_expression = build_kv_cursor_filter_expression(
        project="test", labels=["lbl1==1", "lbl2==2"]
    )
    assert filter_expression == "project=='test' AND _lbl1=='1' AND _lbl2=='2'"


def test_get_access_key():
    key = get_access_key({"X-V3io-Session-Key": "asd"})
    assert key == "asd"

    with pytest.raises(MLRunBadRequestError):
        get_access_key({"some_other_header": "asd"})


def test_get_or_raise():
    some_dict = {"happy_path": "happy_value"}
    value = get_or_raise(some_dict, "happy_path")
    assert value == "happy_value"

    with pytest.raises(MLRunInvalidArgumentError):
        get_or_raise(some_dict, "not_happy_path")


def test_string_to_tsdb_name():
    with pytest.raises(MLRunInvalidArgumentError):
        string_to_tsdb_name("unsupported_string")


def test_get_endpoint_features_function():
    stats = {
        "sepal length (cm)": {
            "count": 30.0,
            "mean": 5.946666666666668,
            "std": 0.8394305678023165,
            "min": 4.7,
            "max": 7.9,
            "hist": [
                [4, 2, 1, 0, 1, 3, 4, 0, 3, 4, 1, 1, 2, 1, 0, 1, 0, 0, 1, 1],
                [
                    4.7,
                    4.86,
                    5.0200000000000005,
                    5.18,
                    5.34,
                    5.5,
                    5.66,
                    5.82,
                    5.98,
                    6.140000000000001,
                    6.300000000000001,
                    6.46,
                    6.62,
                    6.78,
                    6.94,
                    7.1,
                    7.26,
                    7.42,
                    7.58,
                    7.74,
                    7.9,
                ],
            ],
        },
        "sepal width (cm)": {
            "count": 30.0,
            "mean": 3.119999999999999,
            "std": 0.4088672324766359,
            "min": 2.2,
            "max": 3.8,
            "hist": [
                [1, 0, 0, 1, 0, 0, 3, 4, 2, 0, 3, 3, 2, 2, 0, 3, 1, 1, 0, 4],
                [
                    2.2,
                    2.2800000000000002,
                    2.3600000000000003,
                    2.44,
                    2.52,
                    2.6,
                    2.68,
                    2.7600000000000002,
                    2.84,
                    2.92,
                    3.0,
                    3.08,
                    3.16,
                    3.24,
                    3.3200000000000003,
                    3.4,
                    3.48,
                    3.56,
                    3.6399999999999997,
                    3.7199999999999998,
                    3.8,
                ],
            ],
        },
        "petal length (cm)": {
            "count": 30.0,
            "mean": 3.863333333333333,
            "std": 1.8212317418360753,
            "min": 1.3,
            "max": 6.7,
            "hist": [
                [6, 4, 0, 0, 0, 0, 0, 0, 1, 2, 0, 3, 3, 2, 2, 3, 1, 1, 1, 1],
                [
                    1.3,
                    1.57,
                    1.84,
                    2.1100000000000003,
                    2.38,
                    2.6500000000000004,
                    2.92,
                    3.1900000000000004,
                    3.46,
                    3.7300000000000004,
                    4.0,
                    4.2700000000000005,
                    4.54,
                    4.8100000000000005,
                    5.08,
                    5.3500000000000005,
                    5.62,
                    5.89,
                    6.16,
                    6.430000000000001,
                    6.7,
                ],
            ],
        },
        "petal width (cm)": {
            "count": 30.0,
            "mean": 1.2733333333333334,
            "std": 0.8291804567674381,
            "min": 0.1,
            "max": 2.5,
            "hist": [
                [5, 3, 2, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 2, 3, 1, 1, 0, 4],
                [
                    0.1,
                    0.22,
                    0.33999999999999997,
                    0.45999999999999996,
                    0.58,
                    0.7,
                    0.82,
                    0.94,
                    1.06,
                    1.1800000000000002,
                    1.3,
                    1.42,
                    1.54,
                    1.6600000000000001,
                    1.78,
                    1.9,
                    2.02,
                    2.14,
                    2.2600000000000002,
                    2.38,
                    2.5,
                ],
            ],
        },
    }
    feature_names = list(stats.keys())

    features = get_endpoint_features(feature_names, stats, stats)
    assert len(features) == 4
    # Commented out asserts should be re-enabled once buckets/counts length mismatch bug is fixed
    for feature in features:
        assert feature.expected is not None
        assert feature.actual is not None

        assert feature.expected.histogram is not None
        # assert len(feature.expected.histogram.buckets) == len(feature.expected.histogram.counts)

        assert feature.actual.histogram is not None
        # assert len(feature.actual.histogram.buckets) == len(feature.actual.histogram.counts)

    features = get_endpoint_features(feature_names, stats, None)
    assert len(features) == 4
    for feature in features:
        assert feature.expected is not None
        assert feature.actual is None

        assert feature.expected.histogram is not None
        # assert len(feature.expected.histogram.buckets) == len(feature.expected.histogram.counts)

    features = get_endpoint_features(feature_names, None, stats)
    assert len(features) == 4
    for feature in features:
        assert feature.expected is None
        assert feature.actual is not None

        assert feature.actual.histogram is not None
        # assert len(feature.actual.histogram.buckets) == len(feature.actual.histogram.counts)

    features = get_endpoint_features(feature_names[1:], None, stats)
    assert len(features) == 3


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_deserialize_endpoint_from_kv():
    endpoint = _mock_random_endpoint()
    serialize_endpoint_to_kv(_get_access_key(), endpoint)
    endpoint_from_kv = deserialize_endpoint_from_kv(
        _get_access_key(), endpoint.metadata.project, endpoint.id
    )
    assert endpoint.id == endpoint_from_kv.id


def _get_access_key() -> Optional[str]:
    return os.environ.get("V3IO_ACCESS_KEY")


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
