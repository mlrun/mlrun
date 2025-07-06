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

import pytest
import urllib3.exceptions


@pytest.mark.integration
def test_ssl_verification_fails(invalid_ssl_ca_k8s_helper):
    with pytest.raises(urllib3.exceptions.MaxRetryError) as exc:
        invalid_ssl_ca_k8s_helper.v1api.get_api_resources()

    inner = exc.value.reason
    assert isinstance(inner, urllib3.exceptions.SSLError)
    assert "certificate" in str(inner).lower()


@pytest.mark.integration
def test_ssl_verification_succeeds(valid_k8s_helper):
    pods = valid_k8s_helper.list_pods()
    assert isinstance(pods, list)
