import pathlib

import mlrun
from mlrun import new_function, new_task
from tests.conftest import out_path, tag_test, tests_root_directory, verify_state

from .common import my_func

base_spec = new_task(params={"p1": 8}, out_path=out_path)
input_file_path = str(
    pathlib.Path(__file__).parent / "assets" / "test_run_input_file.txt"
)
base_spec.spec.inputs = {"infile.txt": str(input_file_path)}


def hyper_func(context, p1, p2, p3):
    print(f"p2={p2}, p3={p3}")
    context.log_result("r1", p2 * p3)


def test_handler_hyper():
    run_spec = tag_test(base_spec, "test_handler_hyper")
    run_spec.with_hyper_params({"p1": [1, 5, 3]}, selector="max.accuracy")
    result = new_function().run(run_spec, handler=my_func)
    print(result)
    assert len(result.status.iterations) == 3 + 1, "hyper parameters test failed"
    assert (
        result.status.results["best_iteration"] == 2
    ), "failed to select best iteration"
    verify_state(result)


def test_handler_hyperlist():
    run_spec = tag_test(base_spec, "test_handler_hyperlist")
    run_spec.spec.param_file = f"{tests_root_directory}/param_file.csv"
    result = new_function().run(run_spec, handler=my_func)
    print(result)
    assert len(result.status.iterations) == 3 + 1, "hyper parameters test failed"
    verify_state(result)


def test_hyper_grid():
    grid_params = '{"p2": [2,1,3], "p3": [10,20]}'
    mlrun.datastore.set_in_memory_item("params.json", grid_params)

    run_spec = tag_test(base_spec, "test_hyper_grid")
    run_spec.with_param_file("memory://params.json", selector="r1", strategy="grid")
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    # 3 x p2, 2 x p3 = 6 iterations + 1 header line
    assert len(run.status.iterations) == 1 + 2 * 3, "wrong number of iterations"

    results = [line[5] for line in run.status.iterations[1:]]
    assert results == [20, 10, 30, 40, 20, 60], "unexpected results"
    assert run.output("best_iteration") == 6, "wrong best iteration"


def test_hyper_grid_parallel():
    grid_params = '{"p2": [2,1,3], "p3": [10,20]}'
    mlrun.datastore.set_in_memory_item("params.json", grid_params)

    run_spec = tag_test(base_spec, "test_hyper_grid")
    run_spec.with_param_file(
        "memory://params.json", selector="r1", strategy="grid", parallel_runs=2
    )
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    # 3 x p2, 2 x p3 = 6 iterations + 1 header line
    assert len(run.status.iterations) == 1 + 2 * 3, "wrong number of iterations"


def test_hyper_list():
    list_params = '{"p2": [2,3,1], "p3": [10,30,20]}'
    mlrun.datastore.set_in_memory_item("params.json", list_params)

    run_spec = tag_test(base_spec, "test_hyper_list")
    run_spec.with_param_file("memory://params.json", selector="r1", strategy="list")
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    assert len(run.status.iterations) == 1 + 3, "wrong number of iterations"

    results = [line[5] for line in run.status.iterations[1:]]
    assert results == [20, 90, 20], "unexpected results"
    assert run.output("best_iteration") == 2, "wrong best iteration"


def test_hyper_list_with_stop():
    list_params = '{"p2": [2,3,7,4,5], "p3": [10,10,10,10,10]}'
    mlrun.datastore.set_in_memory_item("params.json", list_params)

    run_spec = tag_test(base_spec, "test_hyper_list_with_stop")
    run_spec.with_param_file(
        "memory://params.json",
        selector="max.r1",
        strategy="list",
        stop_condition="r1>=70",
    )
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    # result: r1 = p2 * p3, r1 >= 70 lead to stop on third run
    assert len(run.status.iterations) == 1 + 3, "wrong number of iterations"
    assert run.output("best_iteration") == 3, "wrong best iteration"


def test_hyper_parallel_with_stop():
    list_params = '{"p2": [2,3,7,4,5], "p3": [10,10,10,10,10]}'
    mlrun.datastore.set_in_memory_item("params.json", list_params)

    run_spec = mlrun.new_task(params={"p1": 1})
    run_spec.with_hyper_params(
        {"p2": [2, 3, 7, 4, 5], "p3": [10, 10, 10, 10, 10]},
        parallel_runs=2,
        selector="max.r1",
        strategy="list",
        stop_condition="r1>=70",
    )
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    # result: r1 = p2 * p3, r1 >= 70 lead to stop on third run
    # may have one extra iterations in flight so checking both 4 or 5
    assert len(run.status.iterations) in [4, 5], "wrong number of iterations"
    assert run.output("best_iteration") == 3, "wrong best iteration"


def test_hyper_random():
    grid_params = {"p2": [2, 1, 3], "p3": [10, 20, 30]}
    run_spec = tag_test(base_spec, "test_hyper_random")
    run_spec.with_hyper_params(
        grid_params, selector="r1", strategy="random", max_iterations=5
    )
    run = new_function().run(run_spec, handler=hyper_func)

    verify_state(run)
    assert len(run.status.iterations) == 1 + 5, "wrong number of iterations"


def custom_hyper_func(context: mlrun.MLClientCtx):
    best_accuracy = 0
    for param in [1, 2, 4, 3]:
        with context.get_child_context(myparam=param) as child:
            accuracy = child.get_param("myparam")
            child.log_result("accuracy", accuracy)
            if accuracy > best_accuracy:
                child.mark_as_best()
                best_accuracy = accuracy


def test_hyper_custom():
    run_spec = tag_test(base_spec, "test_hyper_custom")
    run = new_function().run(run_spec, handler=custom_hyper_func)
    verify_state(run)
    assert len(run.status.iterations) == 1 + 4, "wrong number of iterations"

    results = [line[3] for line in run.status.iterations[1:]]
    print(results)
    assert run.output("best_iteration") == 3, "wrong best iteration"
