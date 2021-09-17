import pytest

import mlrun


def echo(event):
    print(event)
    return event


# tests map, list of (params, request args, expected result)
tests_map = [
    ({"method": "GET"}, {"body": {"x": 5}}, {"get": "ok"}),
    (
        {"method": "POST", "subpath": "data", "return_json": False},
        {"body": b"req text"},
        b"my str",
    ),
    ({"method": "POST", "subpath": "json"}, {"body": {"x": 5}}, {"post": "ok"}),
    (
        {"method": "POST", "subpath": "$path"},
        {"body": {"x": 5}, "path": "/json"},
        {"post": "ok"},
    ),
]


def _new_server(url, engine, method="POST", **kwargs):
    function = mlrun.new_function("test", kind="serving")
    flow = function.set_topology("flow", engine=engine)
    flow.to(name="s1", handler="echo").to(
        "$remote", "remote_echo", url=url, method=method, **kwargs
    ).to(name="s3", handler="echo").respond()
    return function.to_mock_server()


@pytest.mark.parametrize("engine", ["sync", "async"])
def test_remote_step(httpserver, engine):
    httpserver.expect_request("/", method="GET").respond_with_json({"get": "ok"})
    httpserver.expect_request("/foo", method="GET").respond_with_json({"foo": "ok"})
    httpserver.expect_request(
        "/data", method="POST", data="req text"
    ).respond_with_data("my str")
    httpserver.expect_request("/json", method="POST", json={"x": 5}).respond_with_json(
        {"post": "ok"}
    )
    url = httpserver.url_for("/")

    for params, request, expected in tests_map:
        print(f"test params: {params}")
        server = _new_server(url, engine, **params)
        resp = server.test(**request)
        server.wait_for_completion()
        assert resp == expected

    # test with url generated with expression (from the event)
    server = _new_server(None, engine, method="GET", url_expression="event['myurl']")
    resp = server.test(body={"myurl": httpserver.url_for("/foo")})
    server.wait_for_completion()
    assert resp == {"foo": "ok"}
