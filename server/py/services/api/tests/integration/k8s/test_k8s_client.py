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
import kubernetes
import pytest
import urllib3

import services.api.tests.integration.k8s.conftest


@pytest.mark.integration
def test_k8shelper_ssl_verification_fails(bad_ca_kubeconfig_path: str) -> None:
    helper = services.api.tests.integration.k8s.conftest._k8shelper_from_config(
        bad_ca_kubeconfig_path
    )

    with pytest.raises(kubernetes.client.exceptions.ApiException) as exc:
        helper.v1api.get_api_resources()
        assert exc.value.status == 409
        assert "SSLError" in exc.value.reason
        assert isinstance(exc.value.__cause__, urllib3.exceptions.SSLError)


@pytest.mark.integration
def test_k8shelper_ssl_verification_succeeds(valid_kubeconfig_path: str) -> None:
    helper = services.api.tests.integration.k8s.conftest._k8shelper_from_config(
        valid_kubeconfig_path
    )

    pods = helper.list_pods()
    assert isinstance(pods, list)
