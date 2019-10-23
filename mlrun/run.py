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
import socket
from base64 import b64decode
from copy import deepcopy
from os import environ, path, makedirs
from tempfile import mktemp

import yaml

from .execution import MLClientCtx
from .model import RunObject
from .runtimes import (HandlerRuntime, LocalRuntime, RemoteRuntime,
                       DaskCluster, MpiRuntime, KubejobRuntime, SparkRuntime)
from .utils import update_in, get_in, logger
from .datastore import get_object


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
    print('file: {}'.format(context.get_input('infile.txt').get()))

    # RUN some useful code e.g. ML training, data prep, etc.

    # log scalar result values (job result metrics)
    context.log_result('accuracy', p1 * 2)
    context.log_result('loss', p1 * 3)
    context.set_label('framework', 'sklearn')

    # log various types of artifacts (file, web page, table), will be versioned and visible in the UI
    context.log_artifact('model.txt', body=b'abc is 123', labels={'framework': 'xgboost'})
    context.log_artifact('results.html', body=b'<b> Some HTML <b>', viewer='web-app')

    """

    if 'mlrun_context' in globals() and not spec and not event:
        return globals().get('mlrun_context')

    newspec = {}
    config = environ.get('MLRUN_EXEC_CONFIG')
    if event:
        newspec = event.body
        uid = uid or event.id

    elif spec:
        newspec = deepcopy(spec)

    elif with_env and config:
        newspec = config

    if isinstance(newspec, RunObject):
        newspec = newspec.to_dict()

    if newspec and not isinstance(newspec, dict):
        newspec = json.loads(newspec)

    if not newspec:
        newspec = {}

    update_in(newspec, 'metadata.name', name, replace=False)
    autocommit = False
    tmp = environ.get('MLRUN_META_TMPFILE')
    out = environ.get('MLRUN_DBPATH', rundb)
    if out:
        autocommit = True
        logger.info('logging run results to: {}'.format(out))

    ctx = MLClientCtx.from_dict(newspec, rundb=out, autocommit=autocommit,
                                tmp=tmp, host=socket.gethostname())
    return ctx


runtime_dict = {'remote': RemoteRuntime,
                'nuclio': RemoteRuntime,
                'dask': DaskCluster,
                'job': KubejobRuntime,
                'mpijob': MpiRuntime,
                'spark': SparkRuntime}


def import_function(url, name='', project: str = '', tag: str = '',
                    secrets=None):
    runtime = import_function_to_dict(url, secrets)
    return new_function(name, project=project, tag=tag, runtime=runtime)


def import_function_to_dict(url, secrets=None):
    """Load function spec from local/remote YAML file"""
    obj = get_object(url, secrets)
    runtime = yaml.load(obj, Loader=yaml.FullLoader)
    remote = '://' in url

    code = get_in(runtime, 'spec.build.functionSourceCode')
    cmd = code_file = get_in(runtime, 'spec.command', '')
    if ' ' in cmd:
        code_file = cmd[:cmd.find(' ')]
    if runtime['kind'] in ['', 'local'] and code:
        if code:
            fpath = mktemp('.py')
            code = b64decode(code).decode('utf-8')
            update_in(runtime, 'spec.command', fpath)
            with open(fpath, 'w') as fp:
                fp.write(code)
        elif remote and cmd:
            if cmd.startswith('/'):
                raise ValueError('exec path (spec.command) must be relative')
            url = url[:url.rfind('/')+1] + code_file
            code = get_object(url, secrets)
            dir = path.dirname(code_file)
            if dir:
                makedirs(dir, exist_ok=True)
            with open(code_file, 'w') as fp:
                fp.write(code)
        elif cmd:
            if not path.isfile(code_file):
                # look for the file in a relative path to the yaml
                slash = url.rfind('/')
                if slash >= 0 and path.isfile(url[:url.rfind('/') + 1] + code_file):
                    raise ValueError('exec file spec.command={}'.format(code_file) +
                                     ' is relative, change working dir')
                raise ValueError('no file in exec path (spec.command={})'.format(code_file))
        else:
            raise ValueError('command or code not specified in function spec')

    return runtime


def new_function(name: str = '', project: str = '', tag: str = '',
                 command: str = '', image: str = '',
                 runtime=None, args: list = None,
                 mode=None, kfp=None, interactive=False):
    """Create a new ML function from base properties

    e.g.:
           # define a container based function
           f = new_function(command='job://training.py -v', image='myrepo/image:latest')

           # define a handler function (execute a local function handler)
           f = new_function().run(task, handler=myfunction)

    :param name:     function name
    :param project:  function project (none for 'default')
    :param tag:      function version tag (none for 'latest')
    :param command:  runtime type + command/url + args (e.g.: mpijob://training.py --verbose)
                     runtime prefixes: None, local, job, spark, dask, mpijob, nuclio
    :param args:     command line arguments (override the ones in command)
    :param image:    default container image
    :param runtime:  runtime (job, nuclio, spark, dask ..) object/dict
                     store runtime specific details and preferences
    :param rundb:    optional, path/url to the metadata and artifact database
    :param mode:     runtime mode, e.g. noctx, pass to bypass mlrun
    :param kfp:      flag indicating running within kubeflow pipeline
    :param interactive:   run the tasks synchronously and print the output

    :return: function object
    """
    kind, runtime = process_runtime(command, runtime)

    if not kind and not get_in(runtime, 'spec.command', command):
        runner = HandlerRuntime()
    else:
        if kind in ['', 'local'] and get_in(runtime, 'spec.command'):
            runner = LocalRuntime.from_dict(runtime)
        elif kind in runtime_dict:
            runner = runtime_dict[kind].from_dict(runtime)
        else:
            raise Exception('unsupported runtime ({}) or missing command, '.format(kind)
                            + 'supported runtimes: {}'.format(
                              ','.join(list(runtime_dict.keys()) + ['local'])))

    if name:
        runner.metadata.name = name
    if project:
        runner.metadata.project = project
    if tag:
        runner.metadata.tag = tag
    if image:
        runner.spec.image = image
    if args:
        runner.spec.args = args
    runner.kfp = kfp
    runner.spec.mode = mode
    runner.interactive = interactive
    return runner


def process_runtime(command, runtime):
    kind = ''
    if runtime and hasattr(runtime, 'to_dict'):
        runtime = runtime.to_dict()
    if runtime and isinstance(runtime, dict):
        kind = runtime.get('kind', '')
        command = command or get_in(runtime, 'spec.command', '')
    kind, command = get_kind(kind, command or '')
    if not runtime:
        runtime = {}
    update_in(runtime, 'spec.command', command)
    runtime['kind'] = kind
    if kind != 'remote':
        parse_command(runtime, command)
    return kind, runtime


def get_kind(kind, command):
    idx = command.find('://')
    if idx < 0:
        return kind, command
    if command.startswith('http'):
        return 'remote', command
    return command[:idx], command[idx + 3:]


def parse_command(runtime, url):
    idx = url.find('#')
    if idx > -1:
        update_in(runtime, 'spec.image', url[:idx])
        url = url[idx+1:]

    if url:
        arg_list = url.split()
        update_in(runtime, 'spec.command', arg_list[0])
        update_in(runtime, 'spec.args', arg_list[1:])


def code_to_function(name='', filename='', handler='', runtime=None,
                     image=None):
    """convert code or notebook to function object with embedded code
    code stored in the function spec and can be refreshed using .with_code()
    eliminate the need to build container images everytime we edit the code

    :param name:      function name
    :param filename:  blank for current notebook, or path to .py/.ipynb file
    :param handler:   name of function handler (if not main)
    :param runtime:   optional, runtime type local, job, dask, mpijob, ..
    :param image:     optional, container image

    :return:
           function object
    """
    if runtime == 'nuclio':
        r = RemoteRuntime()
        r.metadata.name = name
        return r

    from nuclio import build_file
    bname, spec, code = build_file(filename, handler=handler)

    if runtime is None or runtime in ['', 'local']:
        r = LocalRuntime()
    elif runtime in runtime_dict:
        r = runtime_dict[runtime]()
    else:
        raise Exception('unsupported runtime ({})'.format(runtime))

    h = get_in(spec, 'spec.handler', '').split(':')
    r.handler = h[0] if len(h) <= 1 else h[1]
    r.metadata = get_in(spec, 'spec.metadata')
    r.metadata.name = name or bname or 'mlrun'
    r.spec.image = get_in(spec, 'spec.image', image)
    build = r.spec.build
    build.base_image = get_in(spec, 'spec.build.baseImage')
    build.commands = get_in(spec, 'spec.build.commands')
    build.functionSourceCode = get_in(spec, 'spec.build.functionSourceCode')
    build.image = get_in(spec, 'spec.build.image')
    build.secret = get_in(spec, 'spec.build.secret')
    if r.kind != 'local':
        r.spec.env = get_in(spec, 'spec.env')
        for vol in get_in(spec, 'spec.volumes', []):
            r.spec.volumes.append(vol.get('volume'))
            r.spec.volume_mounts.append(vol.get('volumeMount'))
    return r


