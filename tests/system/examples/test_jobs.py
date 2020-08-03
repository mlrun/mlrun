import kfp.dsl
import kfp.compiler

from mlrun import (
    code_to_function,
    NewTask,
    run_pipeline,
    wait_for_pipeline_completion,
    get_run_db,
)
from mlrun.platforms.other import mount_v3io

from tests.system.base import TestMLRunSystem
from tests.system.examples.base import TestMlRunExamples


@TestMLRunSystem.skip_test_env_not_configured
class TestJobs(TestMlRunExamples):
    def test_run_training_job(self):
        output_path = str(self.results_path / '{{run.uid}}')
        trainer = self._get_trainer()

        self._logger.debug('Creating base task')
        base_task = NewTask(artifact_path=output_path).set_label('stage', 'dev')

        # run our training task, with hyper params, and select the one with max accuracy
        self._logger.debug('Running task with hyper params')
        train_task = NewTask(
            name='my-training', handler='training', params={'p1': 9}, base=base_task
        )
        train_run = trainer.run(train_task)

        # running validation, use the model result from the previous step
        self._logger.debug('Running validation using the model from the previous step')
        model = train_run.outputs['mymodel']
        trainer.run(base_task, handler='validation', inputs={'model': model})

    def test_run_kubeflow_pipeline(self):
        trainer = self._get_trainer()

        @kfp.dsl.pipeline(name='job test', description='demonstrating mlrun usage')
        def job_pipeline(p1: int = 9) -> None:
            """Define our pipeline.

            :param p1: A model parameter.
            """

            train = trainer.as_step(
                handler='training', params={'p1': p1}, outputs=['mymodel']
            )

            trainer.as_step(
                handler='validation',
                inputs={'model': train.outputs['mymodel']},
                outputs=['validation'],
            )

        kfp.compiler.Compiler().compile(job_pipeline, 'jobpipe.yaml')
        artifact_path = 'v3io:///users/admin/kfp/{{workflow.uid}}/'
        arguments = {'p1': 8}
        workflow_run_id = run_pipeline(
            job_pipeline, arguments, experiment='my-job', artifact_path=artifact_path
        )

        wait_for_pipeline_completion(workflow_run_id)
        db = get_run_db().connect()
        self._logger.debug('run', run=db.list_runs(project='default', labels=f'workflow={workflow_run_id}'))

    def _get_trainer(self):
        code_path = str(self.artifacts_path / 'jobs_function.py')

        self._logger.debug('Creating trainer job')
        trainer = code_to_function(name='my-trainer', kind='job', filename=code_path)
        trainer.spec.build.commands.append('pip install pandas')
        trainer.spec.build.base_image = 'mlrun/mlrun'
        trainer.spec.image_pull_policy = 'Always'
        trainer.spec.command = code_path
        trainer.apply(mount_v3io())

        self._logger.debug('Deploying trainer')
        trainer.deploy()

        return trainer
