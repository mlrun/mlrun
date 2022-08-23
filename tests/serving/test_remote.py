# Copyright 2018 Iguazio
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
import re
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


def test_remote_class_to_dict(httpserver):
    from mlrun.serving.remote import RemoteStep

    url = httpserver.url_for("/cat")
    step = RemoteStep(
        name="remote_echo",
        url=url,
        method="GET",
        input_path="req",
        result_path="resp",
        max_in_flight=1,
    )
    assert step.to_dict() == {
        "class_args": {
            "max_in_flight": 1,
            "method": "GET",
            "retries": 6,
            "return_json": True,
            "url": url,
        },
        "class_name": "mlrun.serving.remote.RemoteStep",
        "input_path": "req",
        "name": "remote_echo",
        "result_path": "resp",
    }


# ML-1394
@pytest.mark.parametrize("engine", ["sync", "async"])
def test_remote_class_no_header_propagation(httpserver, engine):
    from mlrun.serving.remote import RemoteStep

    httpserver.expect_request(
        "/cat", method="GET", headers={"X-dont-propagate": "me"}
    ).respond_with_json({"cat": "ok"})

    function = mlrun.new_function("test2", kind="serving")
    flow = function.set_topology("flow", engine=engine)
    flow.to(name="s1", handler="echo").to(
        RemoteStep(
            name="remote_echo",
            url=httpserver.url_for("/cat"),
            method="GET",
            input_path="req",
            result_path="resp",
            retries=0,
        )
    ).to(name="s3", handler="echo").respond()

    server = function.to_mock_server()
    try:
        server.test(body={"req": {"x": 5}}, headers={"X-dont-propagate": "me"})
        assert False
    except RuntimeError:
        pass
    finally:
        try:
            server.wait_for_completion()
        except RuntimeError:
            pass


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
    assert resp == {"req": {"url": "/dog", "data": {"x": 5}}, "resp": {"post": "ok"}}


def _timed_out_handler(request: Request):
    time.sleep(2)  # this should be greater than the client's timeout parameter
    return Response(request.data, status=200)


@pytest.mark.parametrize("engine", ["async", "sync"])
def test_timeout(httpserver, engine):
    httpserver.expect_request("/data", method="POST").respond_with_handler(
        _timed_out_handler
    )
    url = httpserver.url_for("/data")
    server = _new_server(url, engine, timeout=1, retries=0, return_json=False)

    try:
        server.test(body=b"tst", method="POST")
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
        # ignore the delayed errors
        pass


class RetryTester:
    def __init__(self, ok_after=0):
        self.retries_dict = {}
        self.ok_after = ok_after

    def handler(self, request: Request):
        retries = self.retries_dict.get(request.path, 0)
        self.retries_dict[request.path] = retries + 1
        if self.ok_after and retries >= self.ok_after:
            return Response(request.data, status=200)
        return Response("Failed", status=500)


@pytest.mark.parametrize("engine", ["sync", "async"])
def test_failure(httpserver, engine):
    tester = RetryTester()
    method = "POST"
    httpserver.expect_request("/data", method=method).respond_with_handler(
        tester.handler
    )
    url = httpserver.url_for("/data")
    server = _new_server(url, engine, method=method, return_json=False, retries=0)

    try:
        server.test(body=b"tst", method=method)
        assert False, "did not fail the request"
    except RuntimeError:
        pass

    assert tester.retries_dict["/data"] == 1, "did not get expected number of retries"
    try:
        server.wait_for_completion()
    except RuntimeError:
        # ignore the delayed errors
        pass


@pytest.mark.parametrize("engine", ["sync", "async"])
def test_retry(httpserver, engine):
    # test with one failure/retry and 2nd try succeed
    retries = 1
    tester = RetryTester(retries)
    method = "POST"
    httpserver.expect_request("/data", method=method).respond_with_handler(
        tester.handler
    )
    url = httpserver.url_for("/data")
    server = _new_server(url, engine, method=method, return_json=False, retries=retries)
    try:
        server.test(body=b"tst", method=method)
    finally:
        server.wait_for_completion()
    assert (
        tester.retries_dict["/data"] == retries + 1
    ), "did not get expected number of retries"


def _echo_handler(request: Request):
    return Response(request.data, status=200)


def test_parallel_remote(httpserver):
    # test calling multiple http clients
    from mlrun.serving.remote import BatchHttpRequests

    httpserver.expect_request(re.compile("^/.*"), method="POST").respond_with_handler(
        _echo_handler
    )
    url = httpserver.url_for("/")

    function = mlrun.new_function("test2", kind="serving")
    flow = function.set_topology("flow", engine="async")
    flow.to(
        BatchHttpRequests(
            url_expression="event['url']",
            body_expression="event['data']",
            method="POST",
            input_path="req",
            result_path="resp",
        )
    ).respond()

    server = function.to_mock_server()
    items = list(range(2))
    request = [{"url": f"{url}{i}", "data": i} for i in items]
    try:
        resp = server.test(body={"req": request})
    finally:
        server.wait_for_completion()
    assert resp["resp"] == items, "unexpected response"


def test_parallel_remote_retry(httpserver):
    # test calling multiple http clients with failure and one retry
    from mlrun.serving.remote import BatchHttpRequests

    retries = 1
    tester = RetryTester(retries)
    httpserver.expect_request(re.compile("^/.*"), method="POST").respond_with_handler(
        tester.handler
    )
    url = httpserver.url_for("/")

    function = mlrun.new_function("test2", kind="serving")
    flow = function.set_topology("flow", engine="async")
    flow.to(
        BatchHttpRequests(
            url_expression="event['url']",
            body_expression="event['data']",
            method="POST",
            input_path="req",
            result_path="resp",
            retries=1,
        )
    ).respond()

    server = function.to_mock_server()
    items = list(range(2))
    request = [{"url": f"{url}{i}", "data": i} for i in items]
    try:
        resp = server.test(body={"req": request})
    finally:
        server.wait_for_completion()
    assert resp["resp"] == items, "unexpected response"
    assert tester.retries_dict == {
        "/1": retries + 1,
        "/0": retries + 1,
    }, "didnt retry properly"
