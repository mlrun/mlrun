import os

import mlrun

from tests.system.base import TestMLRunSystem


class TestDemo(TestMLRunSystem):

    project_name = ''

    def custom_setup(self):
        mlrun.set_environment(
            artifact_path='/User/data', project=self.project_name,
        )
        self._artifact_path = os.path.join(
            mlrun.mlconf.artifact_path, 'pipeline/{{workflow.uid}}'
        )
        self._demo_project = self.create_demo_project()

        self._logger.debug(
            'Project Ready',
            project_name=self.project_name,
            project=self._demo_project.to_dict(),
        )

    def custom_teardown(self):
        db = mlrun.get_run_db()

        db.del_runs(project=self.project_name, labels={'kind': 'job'})
        db.del_artifacts(tag='*', project=self.project_name)

    def create_demo_project(self) -> mlrun.projects.MlrunProject:
        raise NotImplementedError

    def run_and_verify_project(self, runs_amount: int = 1, arguments: dict = None):
        arguments = arguments or {}
        run_id = self._demo_project.run(
            'main',
            arguments=arguments,
            artifact_path=self._artifact_path,
            dirty=True,
            watch=True,
        )

        db = mlrun.get_run_db().connect()
        runs = db.list_runs(project=self.project_name, labels=f'workflow={run_id}')

        self._logger.debug('Completed Runs', runs=runs)

        assert len(runs) == runs_amount
        for run in runs:
            assert run['status']['state'] == 'completed'
