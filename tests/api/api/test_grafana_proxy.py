import os
from typing import Optional

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

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


def _get_access_key() -> Optional[str]:
    return os.environ.get("V3IO_ACCESS_KEY")
