import time

import pytest
from werkzeug.wrappers import Request, Response

import mlrun
from mlrun.serving.utils import event_id_key, event_path_key


def echo(event):
    print(event)
    return event


# tests map, list of (params, request args, expected result)
tests_map = [
    ({"method": "GET"}, {"body": {"x": 5}}, {"get": "ok"}),
    (
        {"method": "POST", "subpath": "data", "return_json": False},
        {"path": "/datapath", "body": b"req text", "event_id": "123"},
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
    function = mlrun.new_function("test1", kind="serving")
    flow = function.set_topology("flow", engine=engine)
    flow.to(name="s1", handler="echo").to(
        "$remote", "remote_echo", url=url, method=method, **kwargs
    ).to(name="s3", handler="echo").respond()
    return function.to_mock_server()


@pytest.mark.parametrize("engine", ["sync", "async"])
def test_remote_step(httpserver, engine):
    httpserver.expect_request("/", method="GET").respond_with_json({"get": "ok"})
    httpserver.expect_request("/foo", method="GET").respond_with_json({"foo": "ok"})

    # verify the remote step added headers for the event path and id
    expected_headers = {
        event_id_key: "123",
        event_path_key: "/datapath",
    }
    httpserver.expect_request(
        "/data", method="POST", data="req text", headers=expected_headers
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


@pytest.mark.parametrize("engine", ["async"])
def test_remote_step_bad_status_code(httpserver, engine):
    httpserver.expect_request("/", method="GET").respond_with_json({"get": "ok"})
    httpserver.expect_request("/foo", method="GET").respond_with_json({}, status=400)

    # verify the remote step added headers for the event path and id
    expected_headers = {
        event_id_key: "123",
        event_path_key: "/datapath",
    }
    httpserver.expect_request(
        "/data", method="POST", data="req text", headers=expected_headers
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
    with pytest.raises(RuntimeError):
        server.test(body={"myurl": httpserver.url_for("/foo")})
    with pytest.raises(ValueError):
        server.wait_for_completion()


@pytest.mark.parametrize("engine", ["sync", "async"])
def test_remote_class(httpserver, engine):
    from mlrun.serving.remote import RemoteStep

    httpserver.expect_request("/cat", method="GET").respond_with_json({"cat": "ok"})

    function = mlrun.new_function("test2", kind="serving")
    flow = function.set_topology("flow", engine=engine)
    flow.to(name="s1", handler="echo").to(
        RemoteStep(
            name="remote_echo",
            url=httpserver.url_for("/cat"),
            method="GET",
            input_path="req",
            result_path="resp",
        )
    ).to(name="s3", handler="echo").respond()

    server = function.to_mock_server()
    resp = server.test(body={"req": {"x": 5}})
    server.wait_for_completion()
    assert resp == {"req": {"x": 5}, "resp": {"cat": "ok"}}


@pytest.mark.parametrize("engine", ["sync", "async"])
def test_remote_advance(httpserver, engine):
    from mlrun.serving.remote import RemoteStep

    httpserver.expect_request("/dog", method="POST", json={"x": 5}).respond_with_json(
        {"post": "ok"}
    )

    function = mlrun.new_function("test2", kind="serving")
    flow = function.set_topology("flow", engine=engine)
    flow.to(name="s1", handler="echo").to(
        RemoteStep(
            name="remote_echo",
            url=httpserver.url_for("/"),
            url_expression="endpoint + event['url']",
            body_expression="event['data']",
            input_path="req",
            result_path="resp",
        )
    ).to(name="s3", handler="echo").respond()

    server = function.to_mock_server()
    resp = server.test(body={"req": {"url": "/dog", "data": {"x": 5}}})
    server.wait_for_completion()
    print(resp)
    assert resp == {"req": {"url": "/dog", "data": {"x": 5}}, "resp": {"post": "ok"}}


def sleeping(request: Request):
    print("\nDATA:\n", request.data)
    time.sleep(2)  # this should be greater than the client's timeout parameter
    return Response("Ok", status=200)


@pytest.mark.parametrize("engine", ["sync", "async"])
def test_timeout(httpserver, engine):
    httpserver.expect_request("/data", method="POST").respond_with_handler(sleeping)
    url = httpserver.url_for("/data")
    server = _new_server(url, engine, timeout=1, return_json=False)

    try:
        resp = server.test(body=b"tst", method="POST")
        print(resp)
        assert False, "did not time out"
    except Exception as exc:
        is_timeout = (
            ("timed out" in str(exc))
            or ("CancelledError" in str(exc))
            or ("TimeoutError" in str(exc))
        )
        if not is_timeout:
            raise exc

    try:
        server.wait_for_completion()
    except Exception:
        # handle potential exceptions due to canceled events
        pass


class RetryTester:
    def __init__(self):
        self.retries = 0

    def handler(self, request: Request):
        self.retries += 1
        print(f"retries={self.retries}")
        return Response(str(self.retries), status=500)


@pytest.mark.parametrize("engine", ["sync"])
@pytest.mark.parametrize("retries", [0, 2])
def test_retry(httpserver, engine, retries):
    tester = RetryTester()
    method = "PUT"
    httpserver.expect_request("/data", method=method).respond_with_handler(
        tester.handler
    )
    url = httpserver.url_for("/data")
    server = _new_server(url, engine, method=method, return_json=False, retries=retries)

    try:
        server.test(body=b"tst", method=method)
        assert False, "did not fail the request"
    except Exception:
        pass

    assert tester.retries == retries + 1, "did not get expected number of retries"
