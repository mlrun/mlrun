# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import socket
from ast import literal_eval
from copy import deepcopy
from os import environ
import yaml

from .execution import MLClientCtx
from .render import run_to_html
from .runtimes import HandlerRuntime, LocalRuntime, RemoteRuntime, DaskRuntime, MpiRuntime
from .utils import update_in, get_in


def get_or_create_ctx(name: str,
                      uid: str = '',
                      event=None,
                      spec=None,
                      with_env: bool = True,
                      rundb: str = ''):
    """ called from within the user program to obtain a run context

    the run context is an interface for receiving parameters, data and logging
    run results, the run context is read from the event, spec, or environment
    (in that order), user can also work without a context (local defaults mode)

    all results are automatically stored in the "rundb" or artifact store,
    the path to the rundb can be specified in the call or obtained from env.

    :param name:     run name (will be overridden by context)
    :param uid:      run unique id (will be overridden by context)
    :param event:    function (nuclio Event object)
    :param spec:     dictionary holding run spec
    :param with_env: look for context in environment vars, default True
    :param rundb:    path/url to the metadata and artifact database

    :return: execution context

    Example:

    # load MLRUN runtime context (will be set by the runtime framework e.g. KubeFlow)
    context = get_or_create_ctx('train')

    # get parameters from the runtime context (or use defaults)
    p1 = context.get_param('p1', 1)
    p2 = context.get_param('p2', 'a-string')

    # access input metadata, values, files, and secrets (passwords)
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(context.get_secret('ACCESS_KEY')))
    print('file: {}'.format(context.get_object('infile.txt').get()))

    # RUN some useful code e.g. ML training, data prep, etc.

    # log scalar result values (job result metrics)
    context.log_result('accuracy', p1 * 2)
    context.log_result('loss', p1 * 3)
    context.set_label('framework', 'sklearn')

    # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
    context.log_artifact('model.txt', body=b'abc is 123', labels={'framework': 'xgboost'})
    context.log_artifact('results.html', body=b'<b> Some HTML <b>', viewer='web-app')

    """

    newspec = {}
    config = environ.get('MLRUN_EXEC_CONFIG')
    if event:
        newspec = event.body
        uid = uid or event.id

    elif spec:
        newspec = deepcopy(spec)

    elif with_env and config:
        newspec = config

    if newspec and not isinstance(newspec, dict):
        newspec = yaml.safe_load(newspec)

    if not newspec:
        newspec = {}

    update_in(newspec, 'metadata.name', name, replace=False)
    autocommit = False
    tmp = environ.get('MLRUN_META_TMPFILE')
    out = environ.get('MLRUN_META_DBPATH', rundb)
    if out:
        autocommit = True

    ctx = MLClientCtx.from_dict(newspec, rundb=out, autocommit=autocommit, tmp=tmp)
    ctx.set_label('host', socket.gethostname())
    return ctx


def run_start(struct: dict, command: str = '', args: list = [],
              runtime=None, rundb: str = '', kfp: bool = False,
              handler=None, hyperparams: dict = None,
              param_file: str = None, mode: str = ''):
    """Run a local or remote task.

    :param struct:     dict holding run spec
    :param command:    runtime command (filename, function url, ..)
    :param args:       optional command args
    :param runtime:    runtime spec (dict or object) e.g. MpiJob, Dask, ..
                       provide runtime specific configuration.
    :param rundb:      path/url to the metadata and artifact database
    :param kfp:        flag indicating run within kubeflow pipeline
    :param handler:    pointer or name of a function handler
    :param hyperparams:  hyper parameters (for running multiple iterations)
    :param param_file:   path/url to csv table per run with parameter values
    :param mode:       special run mode, e.g. 'noctx'

    :return: run context object (dict) with run metadata, results and status
    """

    if struct:
        struct = deepcopy(struct)

    if not runtime and handler:
        runtime = HandlerRuntime(handler=handler)
    else:
        if runtime:
            if isinstance(runtime, str):
                runtime = literal_eval(runtime)
            if not isinstance(runtime, dict):
                runtime = runtime.to_dict()

        if not struct:
            struct = {}
        runtime_spec = get_in(struct, 'spec.runtime', runtime or {})

        if command:
            update_in(runtime_spec, 'command', command)
        if args:
            update_in(runtime_spec, 'args', args)

        kind = runtime_spec.get('kind', '')
        command = runtime_spec.get('command', command)
        update_in(struct, 'spec.runtime', runtime_spec)

        if kind == 'remote' or (kind == '' and '://' in command):
            runtime = RemoteRuntime()
        elif kind in ['', 'local'] and command:
            runtime = LocalRuntime()
        elif kind == 'mpijob':
            runtime = MpiRuntime()
        elif kind == 'dask':
            runtime = DaskRuntime()
        else:
            raise Exception('unsupported runtime (%s) or missing command' % kind)

    runtime.handler = handler
    runtime.process_struct(struct, rundb, hyperparams, param_file, mode)
    runtime.with_kfp = kfp

    results = runtime.run()

    return results


def mlrun_op(name: str = '', project: str = '',
             image: str = 'v3io/mlrun', runtime: str = '', command: str = '',
             secrets: list = [], params: dict = {}, hyperparams: dict = {},
             param_file: str = '', inputs: dict = {}, outputs: dict = {},
             in_path: str = '', out_path: str = '', rundb: str = '',
             mode: str = ''):
    """mlrun KubeFlow pipelines operator, use to form pipeline steps

    when using kubeflow pipelines, each step is wrapped in an mlrun_op
    one step can pass state and data to the next step, see example below.

    :param name:    name used for the step
    :param project: optional, project name
    :param image:   optional, run container image (will be executing the step)
                    the container should host all requiered packages + code
                    for the run, alternatively user can mount packages/code via
                    shared file volumes like v3io (see example below)
    :param runtime: optional, runtime specification
    :param command: exec command (or URL for functions)
    :param secrets: extra secrets specs, will be injected into the runtime
                    e.g. ['file=<filename>', 'env=ENV_KEY1,ENV_KEY2']
    :param params:  dictionary of run parameters and values
    :param hyperparams: dictionary of hyper parameters and list values, each
                        hyperparam holds a list of values, the run will be
                        executed for every parameter combination (GridSearch)
    :param param_file:  a csv file with parameter combinations, first row hold
                        the parameter names, following rows hold param values
    :param inputs:   dictionary of input objects + optional paths (if path is
                     omitted the path will be the in_path/key.
    :param outputs:  dictionary of input objects + optional paths (if path is
                     omitted the path will be the out_path/key.
    :param in_path:  default input path/url (prefix) for inputs
    :param out_path: default output path/url (prefix) for artifacts
    :param rundb:    path for rundb (or use 'MLRUN_META_DBPATH' env instead)
    :param mode:     run mode, e.g. 'noctx' for pushing params as args

    :return: KFP step operation

    Example:
    from kfp import dsl
    from mlrun import mlrun_op
    from mlrun.platforms import mount_v3io

    def mlrun_train(p1, p2):
    return mlrun_op('training',
                    command = '/User/kubeflow/training.py',
                    params = {'p1':p1, 'p2':p2},
                    outputs = {'model.txt':'', 'dataset.csv':''},
                    out_path ='v3io:///bigdata/mlrun/{{workflow.uid}}/',
                    rundb = '/User/kubeflow')

    # use data from the first step
    def mlrun_validate(modelfile):
        return mlrun_op('validation',
                    command = '/User/kubeflow/validation.py',
                    inputs = {'model.txt':modelfile},
                    out_path ='v3io:///bigdata/mlrun/{{workflow.uid}}/',
                    rundb = '/User/kubeflow')

    @dsl.pipeline(
        name='My MLRUN pipeline', description='Shows how to use mlrun.'
    )
    def mlrun_pipeline(
        p1 = 5 , p2 = '"text"'
    ):
        # run training, mount_v3io will mount "/User" into the pipeline step
        train = mlrun_train(p1, p2).apply(mount_v3io())

        # feed 1st step results into the secound step
        validate = mlrun_validate(train.outputs['model-txt']).apply(mount_v3io())

    """
    from kfp import dsl
    from os import environ

    rundb = rundb or environ.get('MLRUN_META_DBPATH')
    cmd = ['python', '-m', 'mlrun', 'run', '--kfp', '--workflow', '{{workflow.uid}}', '--name', name]
    file_outputs = {}
    for s in secrets:
        cmd += ['-s', f'{s}']
    for p, val in params.items():
        cmd += ['-p', f'{p}={val}']
    for x, val in hyperparams.items():
        cmd += ['-x', f'{x}={val}']
    for i, val in inputs.items():
        cmd += ['-i', f'{i}={val}']
    for o, val in outputs.items():
        cmd += ['-o', f'{o}={val}']
        file_outputs[o.replace('.', '-')] = f'/tmp/{o}'
    if project:
        cmd += ['--project', project]
    if runtime:
        cmd += ['--runtime', runtime]
    if in_path:
        cmd += ['--in-path', in_path]
    if out_path:
        cmd += ['--out-path', out_path]
    if rundb:
        cmd += ['--rundb', rundb]
    if param_file:
        cmd += ['--param-file', param_file]
    if mode:
        cmd += ['--mode', mode]

    if hyperparams or param_file:
        file_outputs['iterations'] = f'/tmp/iteration_results.csv'

    cop = dsl.ContainerOp(
        name=name,
        image=image,
        command=cmd + [command],
        file_outputs=file_outputs,
    )
    return cop
