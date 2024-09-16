# Copyright 2024 Iguazio
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
#
import pytest

import mlrun.common.schemas
import mlrun.runtimes.nuclio


@pytest.mark.parametrize(
    "host, path, expected_url",
    [
        ("example.com", "/api", "example.com/api"),
        ("example.com/", "/api", "example.com/api"),
        ("example.com", "api", "example.com/api"),
        ("example.com/", "api", "example.com/api"),
        ("example.com/", "/api/", "example.com/api"),
        ("example.com/", "/api/long/path/", "example.com/api/long/path"),
        ("example.com", "/", "example.com"),
        ("example.com/", "/", "example.com"),
        ("example.com", None, "example.com"),
        ("example.com/", None, "example.com"),
    ],
)
def test_get_invoke_url(host, path, expected_url):
    # testing server side api gateway
    api_gateway = mlrun.common.schemas.APIGateway(
        metadata=mlrun.common.schemas.APIGatewayMetadata(name="test"),
        spec=mlrun.common.schemas.APIGatewaySpec(
            name="test", host=host, path=path, upstreams=[]
        ),
    )
    assert api_gateway.get_invoke_url() == expected_url

    # testing client side api gateway
    api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway(
        metadata=mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(name="test"),
        spec=mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
            project="test", host=host, path=path, functions=["test"]
        ),
    )
    assert api_gateway.invoke_url == "https://" + expected_url


def test_with_annotations():
    annotations = {"key1": "value1", "key2": "value2"}

    api_gateway = mlrun.runtimes.nuclio.api_gateway.APIGateway(
        metadata=mlrun.runtimes.nuclio.api_gateway.APIGatewayMetadata(name="test"),
        spec=mlrun.runtimes.nuclio.api_gateway.APIGatewaySpec(
            project="test", host="host", path="path", functions=["test"]
        ),
    )

    api_gateway.with_annotations(annotations)
    assert api_gateway.metadata.annotations == annotations
