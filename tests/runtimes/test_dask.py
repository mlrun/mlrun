# Copyright 2023 Iguazio
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

import mlrun
import mlrun.errors
from mlrun import new_function, new_task
from mlrun.common.types import AuthenticationMode
from tests.conftest import tag_test, verify_state

has_dask = False
try:
    import dask  # noqa

    has_dask = True
except ImportError:
    pass


def inc(x):
    return x + 2


def my_func(context, p1=1, p2="a-string"):
    print(f"Run: {context.name} (uid={context.uid})")
    print(f"Params: p1={p1}, p2={p2}\n")

    x = context.dask_client.submit(inc, p1)

    context.log_result("accuracy", x.result())
    context.log_artifact("chart", body="abc")
    return f"tst-me-{context.iteration}"


@pytest.mark.skipif(not has_dask, reason="missing dask")
def test_dask_local():
    spec = tag_test(new_task(params={"p1": 3, "p2": "vv"}), "test_dask_local")
    function = new_function(kind="dask")
    function.spec.remote = False
    run = function.run(spec, handler=my_func)
    verify_state(run)


@pytest.mark.parametrize(
    "auth_mode",
    [
        AuthenticationMode.NONE,
        AuthenticationMode.BASIC,
        AuthenticationMode.BEARER,
        AuthenticationMode.IGUAZIO,
    ],
)
def test_dask_validate_passes_when_not_iguazio_v4(monkeypatch, auth_mode):
    monkeypatch.setattr(mlrun.mlconf.httpdb.authentication, "mode", auth_mode)

    # Dask is supported outside IG4 - validate() must not raise
    new_function(kind="dask").validate()


def test_dask_validate_raises_in_iguazio_v4(monkeypatch):
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication, "mode", AuthenticationMode.IGUAZIO_V4
    )

    with pytest.raises(
        mlrun.errors.MLRunBadRequestError, match="not supported on this system"
    ):
        new_function(kind="dask").validate()


def test_dask_run_raises_in_iguazio_v4(monkeypatch):
    # Dask runs via the client-side local launcher (Dask is _is_remote=False), which now calls
    # runtime.validate() in _validate_run. On IG4 run() must fail fast with the clear error
    # instead of failing late while bringing up the cluster.
    monkeypatch.setattr(
        mlrun.mlconf.httpdb.authentication, "mode", AuthenticationMode.IGUAZIO_V4
    )

    with pytest.raises(
        mlrun.errors.MLRunBadRequestError, match="not supported on this system"
    ):
        new_function(kind="dask").run(handler=my_func)
