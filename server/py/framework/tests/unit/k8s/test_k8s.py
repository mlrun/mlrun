# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
from dataclasses import dataclass
from typing import Optional

import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, ReadTimeoutError

from framework.utils.singletons.k8s import K8sHelper


@pytest.fixture(autouse=True)
def patch_kube_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "framework.utils.singletons.k8s.config.load_incluster_config",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "framework.utils.singletons.k8s.config.load_kube_config",
        lambda *_, **__: None,
    )


class MockOKResponse:
    msg = HTTPHeaderDict({"Content-Type": "application/json"})
    status = 200
    reason = "OK"
    version = 11

    def __init__(self) -> None:
        self._fp = io.BytesIO(b'{"all" : "good"}')

    def read(self, _: Optional[int] = None) -> bytes:
        return self._fp.read()

    def isclosed(self) -> bool:
        return False

    def getheader(
        self,
        name: str,
        default: Optional[str] = None,
    ):
        return self.msg.get(name, default)

    def getheaders(self):
        return list(self.msg.items())


@dataclass
class AttemptCounter:
    count: int = 0

    def inc(self) -> int:
        self.count += 1
        return self.count


def test_retry_config_values() -> None:
    retries = K8sHelper()._api_config.retries
    assert (retries.connect, retries.read) == (3, 3)
    assert {"GET", "HEAD"} <= set(retries.allowed_methods)


@pytest.mark.parametrize(
    "exc_cls",
    [
        ConnectTimeoutError,
        ReadTimeoutError,
    ],
)
def test_k8s_helper_retries_on_errors(
    monkeypatch: pytest.MonkeyPatch,
    exc_cls: type[Exception],
) -> None:
    helper = K8sHelper()
    attempts = AttemptCounter()

    def patched_make_request(
        self,
        conn,
        method,
        url,
        *a,
        **kw,
    ):
        if attempts.inc() == 1:
            raise exc_cls(None, url, "request failed")
        return MockOKResponse()

    monkeypatch.setattr(
        HTTPConnectionPool,
        "_make_request",
        patched_make_request,
        raising=True,
    )
    monkeypatch.setattr(
        HTTPSConnectionPool,
        "_make_request",
        patched_make_request,
        raising=True,
    )

    resp = helper._api_client.call_api(
        "/api/v1/namespaces",
        "GET",
        response_type="str",
        auth_settings=[],
        _preload_content=False,
        _return_http_data_only=True,
    )

    assert attempts.count == 2
    assert resp.data == b'{"all" : "good"}'
