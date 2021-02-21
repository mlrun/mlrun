import os
from typing import Optional

import pytest
from fastapi.testclient import TestClient
from pytest import fail
from sqlalchemy.orm import Session
from v3io.dataplane import RaiseForStatus
from v3io_frames import CreateError
from v3io_frames import frames_pb2 as fpb2

from mlrun.api.api.endpoints.grafana_proxy import (
    _parse_query_parameters,
    _validate_query_parameters,
)
from mlrun.api.api.endpoints.model_endpoints import (
    ENDPOINTS_TABLE_PATH,
    ENDPOINT_EVENTS_TABLE_PATH,
)
from mlrun.config import config
from mlrun.errors import MLRunBadRequestError
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client
from tests.api.api.test_model_endpoints import (
    _mock_random_endpoint,
    _write_endpoint_to_kv,
)

ENV_PARAMS = {"V3IO_ACCESS_KEY", "V3IO_API", "V3IO_FRAMESD"}


def _build_skip_message():
    return f"One of the required environment params is not initialized ({', '.join(ENV_PARAMS)})"


def _is_env_params_dont_exist() -> bool:
    return not all((os.environ.get(r, False) for r in ENV_PARAMS))


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_grafana_proxy_model_endpoints_check_connection(
    db: Session, client: TestClient
):
    response = client.get(
        url="/api/grafana-proxy/model-endpoints",
        headers={"X-V3io-Session-Key": _get_access_key()},
    )
    assert response.status_code == 200


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_grafana_list_endpoints(db: Session, client: TestClient):
    endpoints_in = [_mock_random_endpoint("active") for _ in range(5)]

    for endpoint in endpoints_in:
        _write_endpoint_to_kv(endpoint)

    response = client.post(
        url="/api/grafana-proxy/model-endpoints/query",
        headers={"X-V3io-Session-Key": _get_access_key()},
        json={"targets": [{"target": "project=test;target_endpoint=list_endpoints"}]},
    )

    response_json = response.json()
    if not response_json:
        fail(f"Empty response, expected list of dictionaries. {response_json}")

    response_json = response_json[0]
    if not response_json:
        fail(
            f"Empty dictionary, expected dictionary with 'columns', 'rows' and 'type' fields. {response_json}"
        )

    if "columns" not in response_json:
        fail(f"Missing 'columns' key in response dictionary. {response_json}")

    if "rows" not in response_json:
        fail(f"Missing 'rows' key in response dictionary. {response_json}")

    if "type" not in response_json:
        fail(f"Missing 'type' key in response dictionary. {response_json}")

    assert len(response_json["rows"]) == 5


def test_parse_query_parameters_should_fail():
    # No 'targets' in body
    with pytest.raises(MLRunBadRequestError):
        _parse_query_parameters({})

    # No 'target' list in 'targets' dictionary
    with pytest.raises(MLRunBadRequestError):
        _parse_query_parameters({"targets": []})

    # Target query not separated by equals ('=') char
    with pytest.raises(MLRunBadRequestError):
        _parse_query_parameters({"targets": [{"target": "test"}]})


def test_parse_query_parameters_should_not_fail():
    # Target query separated by equals ('=') char
    params = _parse_query_parameters({"targets": [{"target": "test=some_test"}]})
    assert params["test"] == "some_test"

    # Target query separated by equals ('=') char (multiple queries)
    params = _parse_query_parameters(
        {"targets": [{"target": "test=some_test;another_test=some_other_test"}]}
    )
    assert params["test"] == "some_test"
    assert params["another_test"] == "some_other_test"


def test_validate_query_parameters_should_fail():
    # No 'target_endpoint' in query parameters
    with pytest.raises(MLRunBadRequestError):
        _validate_query_parameters({})

    # target_endpoint unsupported
    with pytest.raises(MLRunBadRequestError):
        _validate_query_parameters({"target_endpoint": "unsupported_endpoint"})


def test_validate_query_parameters_success():
    _validate_query_parameters({"target_endpoint": "list_endpoints"})


def _get_access_key() -> Optional[str]:
    return os.environ.get("V3IO_ACCESS_KEY")


@pytest.fixture(autouse=True)
def cleanup_endpoints(db: Session, client: TestClient):
    if not _is_env_params_dont_exist():
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
