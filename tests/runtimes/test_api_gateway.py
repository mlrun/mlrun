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
