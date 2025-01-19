import pathlib

import mlrun.config
import mlrun.launcher.remote

import services.api.launcher

assets_path = pathlib.Path(__file__).parent / "assets"
func_path = assets_path / "sample_function.py"
handler = "hello_word"


def test_new_function_args_with_default_image_pull_secret(rundb_mock):
    mlrun.mlconf.function.spec.image_pull_secret = 'adam-docker-registry-auth'
    launcher = services.api.launcher.ServerSideLauncher(
        auth_info=mlrun.common.schemas.AuthInfo()
    )
    runtime = mlrun.code_to_function(
        name="test",
        kind="job",
        filename=str(func_path),
        handler=handler,
        image="mlrun/mlrun",
    )
    uid = "123"
    run = mlrun.run.RunObject(
        metadata=mlrun.model.RunMetadata(uid=uid),
    )
    rundb_mock.store_run(run, uid)
    run = launcher._create_run_object(run)

    run = launcher._enrich_run(
        runtime,
        run=run,
    )
    assert run.spec.image_pull_secret == mlrun.mlconf.function.spec.image_pull_secret
