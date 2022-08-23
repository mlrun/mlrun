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
import datetime

import mlrun
from mlrun.feature_store.steps import SetEventMetadata


def extract_meta(event):
    event.body = {
        "id": event.id,
        "key": event.key,
        "time": event.time,
    }
    return event


def test_set_event_meta():
    function = mlrun.new_function("test1", kind="serving")
    flow = function.set_topology("flow")
    flow.to(SetEventMetadata(id_path="myid", key_path="mykey", time_path="mytime")).to(
        name="e", handler="extract_meta", full_event=True
    ).respond()

    server = function.to_mock_server()
    event = {"myid": "34", "mykey": "123", "mytime": "2022-01-18 15:01"}
    resp = server.test(body=event)
    server.wait_for_completion()
    assert resp == {
        "id": "34",
        "key": "123",
        "time": datetime.datetime(2022, 1, 18, 15, 1),
    }


def test_set_event_random_id():
    function = mlrun.new_function("test2", kind="serving")
    flow = function.set_topology("flow")
    flow.to(SetEventMetadata(random_id=True)).to(
        name="e", handler="extract_meta", full_event=True
    ).respond()

    server = function.to_mock_server()
    resp = server.test(body={"data": "123"}, event_id="XYZ")
    server.wait_for_completion()
    assert resp["id"] != "XYZ", "id was not overwritten"
