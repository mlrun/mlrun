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

import json
import uuid
from ast import literal_eval
from copy import deepcopy
from os import environ
import yaml

from .execution import MLClientCtx
from .runtimes import HandlerRuntime, LocalRuntime, RemoteRuntime, DaskRuntime, MpiRuntime


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
    if with_env and config:
        newspec = config

    elif event:
        newspec = event.body
        uid = uid or event.id

    elif spec:
        newspec = deepcopy(spec)

    if newspec and not isinstance(newspec, dict):
        newspec = yaml.safe_load(newspec)

    if not newspec.get('metadata'):
        newspec['metadata'] = {}

    newspec['metadata']['uid'] = newspec['metadata'].get('uid', uid) or uuid.uuid4().hex
    newspec['metadata']['name'] = newspec['metadata'].get('name', name)

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
    if 'spec' not in struct:
        struct['spec'] = {}

    if not runtime and handler:
        runtime = HandlerRuntime(handler=handler)
    else:
        if runtime:
            if isinstance(runtime, str):
                runtime = literal_eval(runtime)
            if not isinstance(runtime, dict):
                runtime = runtime.to_dict()

            struct['spec']['runtime'] = runtime

        if struct and 'spec' in struct.keys() and 'runtime' in struct['spec'].keys():
            kind = struct['spec']['runtime'].get('kind', '')
            command = struct['spec']['runtime'].get('command', '')
            if kind == 'remote' or (kind == '' and '://' in command):
                runtime = RemoteRuntime()
            elif kind in ['', 'local']:
                runtime = LocalRuntime()
            elif kind == 'mpijob':
                runtime = MpiRuntime()
            elif kind == 'dask':
                runtime = DaskRuntime()
            else:
                raise Exception('unsupported runtime - %s' % kind)

        elif command:
            if '://' in command:
                runtime = RemoteRuntime(command, args)
            else:
                runtime = LocalRuntime(command, args)

        else:
            raise Exception('runtime was not specified via struct or runtime or command!')

    runtime.rundb = rundb
    runtime.handler = handler
    runtime.process_struct(struct)
    runtime.with_kfp = kfp

    return runtime.run(hyperparams)



