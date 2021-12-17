import json

import mlrun
import mlrun.api.schemas
import tests.integration.sdk_api.base


class TestRuns(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_store_big_run(self):
        """
        Sometimes when the run has artifacts (inputs or outputs) their preview is pretty big (but it is limited to some
        size), when we moved to MySQL a run similar to the one this test is storing was failing to be read from the DB
        after insert on _pickle.UnpicklingError: pickle data was truncated
        So we fixed this by changing the BLOB fields to sqlalchemy.dialects.mysql.MEDIUMBLOB
        This test verifies it's working
        """
        project_name = "runs-project"
        mlrun.new_project(project_name)
        uid = "some-uid"
        run_body_path = str(self.assets_path / "big-run.json")
        with open(run_body_path) as run_body_file:
            run_body = json.load(run_body_file)
        mlrun.get_run_db().store_run(run_body, uid, project_name)
        mlrun.get_run_db().read_run(uid, project_name)
