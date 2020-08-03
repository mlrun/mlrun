import os
import pathlib

import kfp
import kfp.compiler

from mlrun import (
    code_to_function,
    mount_v3io,
    NewTask,
    run_pipeline,
    wait_for_pipeline_completion,
    get_run_db,
)

from tests.system.base import TestMLRunSystem
from tests.system.examples.base import TestMlRunExamples


@TestMLRunSystem.skip_test_env_not_configured
class TestDask(TestMlRunExamples):
    def test_dask(self):
        dsf = self._get_dask_function()
        run_object = dsf.run(handler='main', params={'x': 12})
        self._logger.debug('Finished running task', run_object=run_object.to_dict())

        run_uid = run_object.uid()

        assert run_uid is not None
        self._verify_run_metadata(
            run_object.to_dict()['metadata'],
            uid=run_uid,
            name='mydask-main',
            project='default',
            labels={
                'v3io_user': self._test_env['V3IO_USERNAME'],
                'owner': self._test_env['V3IO_USERNAME'],
            },
        )
        self._verify_run_spec(
            run_object.to_dict()['spec'],
            parameters={'x': 12},
            outputs=[],
            output_path='',
            secret_sources=[],
            data_stores=[],
            scrape_metrics=False,
        )

        assert run_object.state() == 'completed'

    def test_run_pipeline(self):
        dsf = self._get_dask_function()

        @kfp.dsl.pipeline(name="dask_pipeline")
        def dask_pipe(x=1, y=10):

            # use_db option will use a function (DB) pointer instead of adding the function spec to the YAML
            dsf.as_step(
                NewTask(handler='main', name='dask_pipeline', params={'x': x, 'y': y}),
                use_db=True,
            )

        kfp.compiler.Compiler().compile(dask_pipe, 'daskpipe.yaml', type_check=False)
        arguments = {'x': 4, 'y': -5}
        artifact_path = '/User/test'
        workflow_run_id = run_pipeline(
            dask_pipe,
            arguments,
            artifact_path=artifact_path,
            run="DaskExamplePipeline",
            experiment="dask pipe",
        )

        wait_for_pipeline_completion(workflow_run_id)
        db = get_run_db().connect()
        runs = db.list_runs(project='default', labels=f'workflow={workflow_run_id}')
        assert len(runs) == 1

        run = runs[0]
        run_uid = run['metadata']['uid']
        self._verify_run_metadata(
            run['metadata'],
            uid=run_uid,
            name='mydask-main',
            project='default',
            labels={
                'v3io_user': self._test_env['V3IO_USERNAME'],
                'owner': self._test_env['V3IO_USERNAME'],
            },
        )
        self._verify_run_spec(
            run['spec'],
            parameters={'x': 4, 'y': -5},
            outputs=['run_id'],
            output_path='/User/test',
            data_stores=[],
        )

        # remove compiled dask.yaml file
        os.remove(str(pathlib.Path(__file__).absolute().parent / 'daskpipe.yaml'))

    def _get_dask_function(self):
        dsf = code_to_function(
            'mydask',
            kind='dask',
            filename=str(self.artifacts_path / 'dask_function.py'),
        ).apply(mount_v3io())

        dsf.spec.image = 'mlrun/ml-models'
        dsf.spec.remote = True
        dsf.spec.replicas = 1
        dsf.spec.service_type = 'NodePort'
        dsf.spec.image_pull_policy = 'Always'
        dsf.spec.command = str(self.artifacts_path / 'dask_function.py')

        return dsf
