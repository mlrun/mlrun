import mlrun


def test_create_application_runtime():
    fn = mlrun.code_to_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )
    assert fn.kind == mlrun.runtimes.RuntimeKinds.application
    assert fn.spec.image == "mlrun/mlrun"
    assert fn.metadata.name == "application-test"
    assert "IyBDb3B5cmlnaHQgMjAyMyBJZ3VhemlvCiM" in fn.spec.build.functionSourceCode
    assert fn.spec.function_handler == "handler:handler"


def test_deploy_application_runtime(rundb_mock):
    fn = mlrun.code_to_function(
        "application-test", kind="application", image="my/web-app:latest"
    )
    fn.deploy()
    assert fn.spec.config["spec.sidecars"] == [
        {
            "image": "my/web-app:latest",
            "name": "application-test-sidecar",
            "ports": [{"containerPort": 8080, "name": "http", "protocol": "TCP"}],
        }
    ]
    assert fn.get_env("SERVING_PORT") == "8080"
    assert fn.spec.image == ""
