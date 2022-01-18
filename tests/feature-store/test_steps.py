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
    function = mlrun.new_function("test2", kind="serving")
    flow = function.set_topology("flow")
    flow.to(SetEventMetadata(id_path="myid", key_path="mykey", time_path="mytime")).to(
        name="e", handler="extract_meta", full_event=True
    ).respond()

    server = function.to_mock_server()
    event = {"myid": "34", "mykey": "123", "mytime": "2022-01-18 15:01"}
    resp = server.test(body=event)
    print(resp)
