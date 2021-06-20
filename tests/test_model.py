import mlrun.api.schemas
import mlrun.runtimes


def test_enum_yaml_dump():
    function = mlrun.new_function("function-name", kind="job")
    function.status.state = mlrun.api.schemas.FunctionState.ready
    print(function.to_yaml())
