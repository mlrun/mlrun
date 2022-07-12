import mlrun
import tests.integration.sdk_api.base


class TestOperations(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_trigger_migrations(self):
        background_task = mlrun.get_run_db().trigger_migrations()
        assert background_task is None
