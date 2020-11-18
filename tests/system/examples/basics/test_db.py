from mlrun import get_run_db, run_local, new_task

from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestDB(TestMLRunSystem):

    project_name = "db-system-test-project"

    def custom_setup(self):
        self._logger.debug("Connecting to database")

        self._logger.debug("Creating dummy task for db queries")

        # {{run.uid}} will be substituted with the run id, so output will be written to different directories per run
        output_path = str(self.results_path / "{{run.uid}}")
        task = (
            new_task(name="demo", params={"p1": 5}, artifact_path=output_path)
            .with_secrets("file", self.assets_path / "secrets.txt")
            .set_label("type", "demo")
        )

        self._logger.debug("Running dummy task")
        run_object = run_local(
            task, command="training.py", workdir=str(self.assets_path)
        )
        self._logger.debug(
            "Finished running dummy task", run_object=run_object.to_dict()
        )

        self._run_uid = run_object.uid()

    def test_db_commands(self):

        # TODO: understand why a single db instantiation isn't enough, and fix the bug in the db
        self._run_db = get_run_db()
        runs = self._run_db.list_runs(project=self.project_name)
        assert len(runs) == 1

        self._verify_run_metadata(
            runs[0]["metadata"],
            uid=self._run_uid,
            name="demo",
            project=self.project_name,
            labels={
                "v3io_user": self._test_env["V3IO_USERNAME"],
                "kind": "",
                "owner": self._test_env["V3IO_USERNAME"],
                "framework": "sklearn",
            },
        )
        self._verify_run_spec(
            runs[0]["spec"],
            parameters={"p1": 5, "p2": "a-string"},
            inputs={"infile.txt": str(self.assets_path / "infile.txt")},
            outputs=[],
            output_path=str(self.results_path / self._run_uid),
            secret_sources=[],
            data_stores=[],
        )

        artifacts = self._run_db.list_artifacts(project=self.project_name)
        assert len(artifacts) == 4
        for artifact_key in ["chart", "html_result", "model", "mydf"]:
            artifact_exists = False
            for artifact in artifacts:
                if artifact["key"] == artifact_key:
                    artifact_exists = True
                    break
            assert artifact_exists

        runtimes = self._run_db.list_runtimes()
        assert len(runtimes) == 4
        for runtime_kind in ["dask", "job", "spark", "mpijob"]:
            runtime_exists = False
            for runtime in runtimes:
                if runtime["kind"] == runtime_kind:
                    runtime_exists = True
                    break
            assert runtime_exists
