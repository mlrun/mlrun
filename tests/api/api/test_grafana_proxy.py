import json
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
        url="/api/projects/grafana-proxy/model-endpoints",
        headers={"X-V3io-Session-Key": _get_access_key()},
    )
    assert response.status_code == 200


def test_grafana_list_endpoints():
    pass


@pytest.mark.skipif(
    _is_env_params_dont_exist(), reason=_build_skip_message(),
)
def test_grafana_list_endpoints(db: Session, client: TestClient):
    endpoints_in = [_mock_random_endpoint("active") for _ in range(5)]

    for endpoint in endpoints_in:
        _write_endpoint_to_kv(endpoint)

    response = client.get(
        url="/api/projects/grafana-proxy/model-endpoints/query",
        headers={"X-V3io-Session-Key": _get_access_key()},
        json=json.dumps({"project": "test"}),
    )

    # TODO: Fix weird pathing issue
    pass
    # endpoints_out = [
    #     ModelEndpoint(**e["endpoint"]) for e in response.json()["endpoints"]
    # ]
    #
    # endpoints_in_set = {json.dumps(e.dict(), sort_keys=True) for e in endpoints_in}
    # endpoints_out_set = {json.dumps(e.dict(), sort_keys=True) for e in endpoints_out}
    # endpoints_intersect = endpoints_in_set.intersection(endpoints_out_set)
    #
    # assert len(endpoints_intersect) == 5


def test_parse_query_parameters_should_fail():
    # No 'targets' in body
    try:
        _parse_query_parameters({})
        fail("Request body did not contain 'targets' key, but did not throw exception")
    except MLRunBadRequestError:
        pass

    # No 'target' list in 'targets' dictionary
    try:
        _parse_query_parameters({"targets": []})
        fail("Request body did not contain 'target' key, but did not throw exception")
    except MLRunBadRequestError:
        pass

    # Target query not separated by equals ('=') char
    try:
        _parse_query_parameters({"targets": [{"target": "test"}]})
        fail("Target query does not contain key values separated by '=' char")
    except MLRunBadRequestError:
        pass

    # Target query not separated by equals ('=') char
    params = _parse_query_parameters({"targets": [{"target": "test=sometest"}]})
    assert params["test"] == "sometest"


def test_parse_query_parameters_should_not_fail():
    # Target query not separated by equals ('=') char
    params = _parse_query_parameters({"targets": [{"target": "test=sometest"}]})
    assert params["test"] == "sometest"


def test_validate_query_parameters_should_fail():
    # No 'target_endpoint' in query parameters
    try:
        _validate_query_parameters({})
        fail(
            "Query parameters do not contain 'target_endpoint', but did not throw exception"
        )
    except MLRunBadRequestError:
        pass

    # target_endpoint unsupported
    try:
        _validate_query_parameters({"target_endpoint": "unsupported_endpoint"})
        fail(
            "Query parameters contains unsupported 'target_endpoint', but did not throw exception"
        )
    except MLRunBadRequestError:
        pass


def test_validate_query_parameters_should_not_fail():
    _validate_query_parameters({"target_endpoint": "list_endpoints"})


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
