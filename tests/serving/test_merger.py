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
import asyncio

import pytest
import storey

import mlrun
from mlrun.serving.merger import Merge


async def double(event):
    return event * 2


class Adder:
    def __init__(self, add=2):
        self.add = add

    def do(self, event):
        return event + self.add


class AsyncAdder(storey.flow._ConcurrentJobExecution):
    def __init__(self, add=2, wait=1, **kwargs):
        super().__init__(**kwargs)
        self.add = add
        self.wait = wait

    async def _process_event(self, event):
        await asyncio.sleep(self.wait)
        return event.body + self.add

    async def _handle_completed(self, event, response):
        new_event = self._user_fn_output_to_event(event, response)
        await self._do_downstream(new_event)


class Gather:
    def __init__(self, context):
        self.context = context
        context.mylist = []

    def do(self, event):
        print("got:", event)
        self.context.mylist.append(event)
        return event


@pytest.mark.parametrize("with_queue", [False, True])
def test_no_merger(with_queue):
    # split and merge without joining events per key/id
    fn = mlrun.new_function("x", kind="serving")
    graph = fn.set_topology("flow", exist_ok=True)
    dbl = graph.to(name="double", handler="double")
    dbl.to(name="add3", class_name="Adder", add=3)
    dbl.to(name="add2", class_name="Adder", add=2)
    if with_queue:
        graph.add_step("$queue", "q1", path="").after_step("add2", "add3").to("Gather")
    else:
        graph.add_step("Gather").after_step("add2", "add3")

    server = fn.to_mock_server()
    for data in [5, 10, 15]:
        server.test("", body=data)
    server.wait_for_completion()
    # expected => [double+2, double+3, ..]  (double = x*2)
    # return 2x number of elements since we dont merge
    assert sorted(server.context.mylist) == [12, 13, 22, 23, 32, 33]


def test_simple():
    # test split and merge and join events by event.id
    fn = mlrun.new_function("x", kind="serving")
    graph = fn.set_topology("flow", exist_ok=True)
    dbl = graph.to(name="double", handler="double")
    dbl.to(name="add3", class_name="Adder", add=3)
    dbl.to(name="add2", class_name="Adder", add=2)
    echo = graph.add_step(Merge(name="Merge")).respond()
    echo.after_step("add2", "add3")

    print(fn.to_yaml())  # verify YAML serialization
    server = fn.to_mock_server()
    resp = server.test("", body=5)
    # expected => [double+2, double+3]  (double = x*2)
    assert sorted(resp) == [12, 13]
    resp = server.test("", body=6)
    assert sorted(resp) == [14, 15]
    server.wait_for_completion()


def test_custom_key():
    # test split and merge and join events by custom key (in event body)
    fn = mlrun.new_function("x", kind="serving")
    graph = fn.set_topology("flow", exist_ok=True)
    dbl = graph.to(name="double", handler="double", input_path="x", result_path="x")
    dbl.to(name="add3", class_name="Adder", add=3, input_path="x", result_path="x")
    dbl.to(name="add2", class_name="Adder", add=2, input_path="x", result_path="x")
    graph.add_step(
        Merge(name="Merge", key_path="event['key']"), after=["add2", "add3"]
    ).respond()

    server = fn.to_mock_server()
    with pytest.raises(RuntimeError) as excinfo:
        server.test("", body={"x": 4})  # missing key
        assert "KeyError" in str(excinfo.value)

    resp = server.test("", body={"x": 4, "key": 77})
    assert sorted([item["x"] for item in resp]) == [10, 11]
    server.wait_for_completion()


def test_delayed():
    # one branch has higher delay leading to events falling behind
    # queue size is limited to 3 events (when it is exceeded old partial events are dropped)
    fn = mlrun.new_function("x", kind="serving")
    graph = fn.set_topology("flow", exist_ok=True)
    dbl = graph.to(name="double", handler="double")
    dbl.to(name="add3", class_name="AsyncAdder", add=3, wait=0.1)
    dbl.to(name="add2", class_name="AsyncAdder", add=2, wait=0.2)

    graph.add_step(Merge(name="Merge", max_behind=3), after=["add2", "add3"]).to(
        "Gather"
    )

    fn.verbose = True
    server = fn.to_mock_server()
    for data in [5, 6, 7, 8, 9]:
        server.test("", body=data)
    server.wait_for_completion()
    mylist = [sorted(item) for item in server.context.mylist]
    assert len(mylist) == 3, "expected 3 results in total (2 were dropped due to delay)"
    assert mylist == [[16, 17], [18, 19], [20, 21]]
