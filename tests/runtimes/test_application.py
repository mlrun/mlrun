import mlrun


def test_create_application_runtime():
    fn = mlrun.code_to_function(
        "application-test", kind="application", image="mlrun/mlrun"
    )
    assert fn.kind == mlrun.runtimes.RuntimeKinds.application
    assert fn.spec.image == "mlrun/mlrun"
    assert fn.metadata.name == "application-test"
    assert (
        "Ly8gQ29weXJpZ2h0IDIwMjMgSWd1YXppbwovLwovLyBMaWN"
        in fn.spec.build.functionSourceCode
    )
    assert fn.spec.function_handler == "reverse_proxy:Handler"


def test_deploy_application_runtime(rundb_mock):
    image = "my/web-app:latest"
    fn = mlrun.code_to_function("application-test", kind="application", image=image)
    fn.deploy()
    assert fn.spec.config["spec.sidecars"] == [
        {
            "image": image,
            "name": "application-test-sidecar",
            "ports": [{"containerPort": 8080, "name": "http", "protocol": "TCP"}],
        }
    ]
    assert fn.get_env("SIDECAR_PORT") == "8080"
    assert fn.status.application_image == image
