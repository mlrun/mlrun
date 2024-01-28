import mlrun


def test_create_deployment_runtime():
    fn = mlrun.code_to_function(
        "deployment-test", kind="deployment", image="mlrun/mlrun"
    )
    assert fn.kind == mlrun.runtimes.RuntimeKinds.deployment
    assert fn.spec.image == "mlrun/mlrun"
    assert fn.metadata.name == "deployment-test"
    assert "IyBDb3B5cmlnaHQgMjAyMyBJZ3VhemlvCiM" in fn.spec.build.functionSourceCode
    assert fn.spec.function_handler == "handler:handler"
