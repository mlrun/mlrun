import mlrun
import tests.integration.sdk_api.base


class TestRun(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_ctx_creation_creates_run_with_project(self):
        ctx_name = "some-context"
        mlrun.get_or_create_ctx(ctx_name)
        runs = mlrun.get_run_db().list_runs(
            name=ctx_name, project=mlrun.mlconf.default_project
        )
        assert len(runs) == 1
        assert runs[0]["metadata"]["project"] == mlrun.mlconf.default_project

    def test_ctx_state_change(self):
        ctx_name = "some-context"
        ctx = mlrun.get_or_create_ctx(ctx_name)
        runs = mlrun.get_run_db().list_runs(
            name=ctx_name, project=mlrun.mlconf.default_project
        )
        assert len(runs) == 1
        assert runs[0]["status"]["state"] == mlrun.runtimes.constants.RunStates.running
        ctx.set_state(mlrun.runtimes.constants.RunStates.completed)
        runs = mlrun.get_run_db().list_runs(
            name=ctx_name, project=mlrun.mlconf.default_project
        )
        assert len(runs) == 1
        assert (
            runs[0]["status"]["state"] == mlrun.runtimes.constants.RunStates.completed
        )
