import kfp.dsl

import mlrun


def func1(context, p1=1):
    context.log_result("accuracy", p1 * 2)


def func2(context, x=0):
    context.log_result("y", x + 1)


@kfp.dsl.pipeline(name="remote_pipeline", description="tests remote pipeline")
def pipeline():
    run1 = mlrun.run_function("func1", handler="func1", params={"p1": 9})
    print(run1)
    run2 = mlrun.run_function("func2", handler="func2", params={"x": 2})
    print(run2)
