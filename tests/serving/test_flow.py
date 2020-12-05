import mlrun


class BaseClass:
    def __init__(self, context, state=None, name=None):
        self.context = context
        self.state = state
        self.name = name


class Echo(BaseClass):
    def do(self, x):
        print("Echo:", self.name, x)
        return x


class EchoError(BaseClass):
    def do(self, x):
        x.body = {'body': x.body, 'origin_state': x.origin_state, 'error': x.error}
        return x


class Chain(BaseClass):
    def do(self, x):
        x.append(self.name)
        return x


class Message(BaseClass):
    def __init__(self, msg="", context=None, name=None):
        self.msg = msg

    def do(self, x):
        print("Messsage:", self.msg)
        return x


class Raiser:
    def __init__(self, msg="", context=None, name=None):
        self.context = context
        self.name = name
        self.msg = msg

    def do(self, x):
        raise ValueError(f" this is an error, {x}")


def test_basic_flow():
    fn = mlrun.new_function("tests", kind="serving")
    fn.set_topology("flow", start_at="s1")
    fn.add_state("s1", class_name="Chain")
    fn.add_state("s2", class_name="Chain", after='$prev')
    fn.add_state("s3", class_name="Chain", after='$prev')

    server = fn.to_mock_server()
    print('\nFlow1:\n', server.graph.to_yaml())
    resp = server.test(body=[])
    assert resp == ['s1', 's2', 's3'], 'flow1 didnt execute properly'

    fn.set_topology("flow", exist_ok=True)
    fn.add_state("s2", class_name="Chain", after='$last')
    fn.add_state("s1", class_name="Chain", after='$start')  # should place s1 first and s2 after it
    fn.add_state("s3", class_name="Chain", after='s2')

    server = fn.to_mock_server()
    print('\nFlow2:\n', server.graph.to_yaml())
    resp = server.test(body=[])
    assert resp == ['s1', 's2', 's3'], 'flow2 didnt execute properly'

    fn.set_topology("flow", exist_ok=True)
    fn.add_state("s1", class_name="Chain", after='$start')
    fn.add_state("s3", class_name="Chain", after='$last')
    fn.add_state("s2", class_name="Chain", after='s1', before='s3')

    server = fn.to_mock_server()
    print('\nFlow3 (insert):\n', server.graph.to_yaml())
    resp = server.test(body=[])
    assert resp == ['s1', 's2', 's3'], 'flow3 didnt execute properly'


def test_on_error():
    fn = mlrun.new_function("tests", kind="serving")
    fn.set_topology("flow", start_at="s1")
    fn.add_state("s1", class_name="Chain")
    fn.add_state("raiser", class_name="Raiser", after='$prev').set_on_error('catch')
    fn.add_state("s3", class_name="Chain", after='$prev')
    fn.add_state("catch", class_name="EchoError").full_event = True

    server = fn.to_mock_server()
    print(server.graph.to_yaml())
    resp = server.test(body=[])
    print(resp)
    assert resp['error'] and resp['origin_state'] == 'raiser', 'error wasnt caught'
