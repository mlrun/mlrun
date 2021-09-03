from mlrun.projects import run_function


def func1(context, p1=1):
    context.log_result("accuracy", p1 * 2)


def func2(context, x=0):
    context.log_result("y", x + 1)


def my_pipe(param1=0):
    run1 = run_function("tstfunc", handler="func1", params={"p1": param1})
    print(run1.to_yaml())

    run2 = run_function(
        "tstfunc", handler="func2", params={"x": run1.outputs["accuracy"]}
    )
    print(run2.to_yaml())
