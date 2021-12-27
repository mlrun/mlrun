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

import pytest

from mlrun import new_function, new_task
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
    context.log_metric("loss", 7)
    context.log_artifact("chart", body="abc")
    return f"tst-me-{context.iteration}"


@pytest.mark.skipif(not has_dask, reason="missing dask")
def test_dask_local():
    spec = tag_test(new_task(params={"p1": 3, "p2": "vv"}), "test_dask_local")
    function = new_function(kind="dask")
    function.spec.remote = False
    run = function.run(spec, handler=my_func)
    verify_state(run)
