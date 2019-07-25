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

from ast import literal_eval
from copy import deepcopy
from os import environ
import yaml

from .execution import MLClientCtx
from .render import run_to_html
from .runtimes import HandlerRuntime, LocalRuntime, RemoteRuntime, DaskRuntime, MpiRuntime
from .utils import update_in, get_in

def get_or_create_ctx(name, uid='', event=None, spec=None, with_env=True, rundb=''):
    """ called from within the user program to obtain a context

    :param name:     run name (will be overridden by context)
    :param uid:      run unique id (will be overridden by context)
    :param event:    function (nuclio Event object)
    :param spec:     dictionary holding run spec
    :param with_env: look for context in environment vars
    :param rundb:    path/url to the metadata and artifact database

    :return: execution context
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

    if not newspec:
        newspec = {}

    if newspec and not isinstance(newspec, dict):
        newspec = yaml.safe_load(newspec)

    update_in(newspec, 'metadata.name', name, replace=False)
    autocommit = False
    tmp = environ.get('MLRUN_META_TMPFILE')
    out = environ.get('MLRUN_META_DBPATH', rundb)
    if out:
        autocommit = True

    ctx = MLClientCtx.from_dict(newspec, rundb=out, autocommit=autocommit, tmp=tmp)
    return ctx


def run_start(struct, command='', args=[], runtime=None, rundb='',
              kfp=False, handler=None, hyperparams=None):
    """Run a local or remote task.

    :param struct:     dict holding run spec
    :param command:    runtime command (filename, function url, ..)
    :param args:       optional command args
    :param runtime:    runtime dict or object or name
    :param rundb:      path/url to the metadata and artifact database
    :param kfp:        flag indicating run within kubeflow pipeline
    :param handler:    pointer or name of a function handler
    :param hyperparams: hyper parameters (for running multiple iterations)

    :return: dict with run metadata and status
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

        runtime_spec = get_in(struct, 'spec.runtime', runtime or {})

        if command:
            update_in(runtime_spec, 'command', command)
        if args:
            update_in(runtime_spec, 'args', args)

        kind = runtime_spec.get('kind', '')
        command = runtime_spec.get('kind', command)
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
    runtime.process_struct(struct, rundb)
    runtime.with_kfp = kfp
    runtime.hyperparams = hyperparams

    results = runtime.run()
    run_to_html(results, True)

    return results


def mlrun_op(name='', project='', image='v3io/mlrun', runtime='', command='', secrets=[],
             params={}, hyperparams={}, inputs={}, outputs={}, out_path='', rundb=''):
    from kfp import dsl

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
    if out_path:
        cmd += ['--out-path', out_path]
    if rundb:
        cmd += ['--rundb', rundb]

    if hyperparams:
        file_outputs['iterations'] = f'/tmp/iterations'

    cop = dsl.ContainerOp(
        name=name,
        image=image,
        command=cmd + [command],
        file_outputs=file_outputs,
    )
    return cop
