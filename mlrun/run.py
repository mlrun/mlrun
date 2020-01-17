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
import uuid
from base64 import b64decode
from copy import deepcopy
from os import environ, makedirs, path
from tempfile import mktemp

import yaml
from nuclio import build_file

from .datastore import get_object
from .db import default_dbpath, get_run_db
from .execution import MLClientCtx
from .funcdoc import find_handlers
from .model import RunObject
from .runtimes import (
    HandlerRuntime, LocalRuntime, RemoteRuntime, runtime_dict
)
from .runtimes.base import EntrypointParam, FunctionEntrypoint
from .runtimes.utils import add_code_metadata
from .utils import get_in, logger, parse_function_uri, update_in
from .config import config as mlconf

def get_or_create_ctx(name: str,
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
    out = rundb or mlconf.dbpath or environ.get('MLRUN_DBPATH')
    if out:
        autocommit = True
        logger.info('logging run results to: {}'.format(out))

    ctx = MLClientCtx.from_dict(newspec, rundb=out, autocommit=autocommit,
                                tmp=tmp, host=socket.gethostname())
    return ctx


def import_function(url='', secrets=None, db=''):
    """create function object from DB or local/remote YAML file

    reading from a file or remote URL (http(s), s3, git, v3io, ..)
    :param url:      path/url to function YAML file
                     or
                     db://{project}/{name}[:tag] when reading from mlrun db

    :param secrets:  optional, credentials dict for DB or URL (s3, v3io, ..)
    :param db        optional, mlrun api/db path

    :return: function object
    """
    if url.startswith('db://'):
        url = url[5:]
        project, name, tag = parse_function_uri(url)
        db = get_run_db(db or default_dbpath()).connect(secrets)
        runtime = db.get_function(name, project, tag)
        if not runtime:
            raise KeyError('function {}:{} not found in the DB'.format(
                name, tag
            ))
        return new_function(runtime=runtime)

    runtime = import_function_to_dict(url, secrets)
    return new_function(runtime=runtime)


def import_function_to_dict(url, secrets=None):
    """Load function spec from local/remote YAML file"""
    obj = get_object(url, secrets)
    runtime = yaml.load(obj, Loader=yaml.FullLoader)
    remote = '://' in url

    code = get_in(runtime, 'spec.build.functionSourceCode')
    update_in(runtime, 'metadata.build.code_origin', url)
    cmd = code_file = get_in(runtime, 'spec.command', '')
    if ' ' in cmd:
        code_file = cmd[:cmd.find(' ')]
    if runtime['kind'] in ['', 'local']:
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
            with open(code_file, 'wb') as fp:
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
                 kind: str = '', command: str = '', image: str = '',
                 args: list = None, runtime=None,
                 mode=None, kfp=None):
    """Create a new ML function from base properties

    e.g.:
           # define a container based function
           f = new_function(command='job://training.py -v', image='myrepo/image:latest')

           # define a handler function (execute a local function handler)
           f = new_function().run(task, handler=myfunction)

    :param name:     function name
    :param project:  function project (none for 'default')
    :param tag:      function version tag (none for 'latest')

    :param kind:     runtime type (local, job, nuclio, spark, mpijob, dask, ..)
    :param command:  command/url + args (e.g.: training.py --verbose)
    :param image:    container image (start with '.' for default registry)
    :param args:     command line arguments (override the ones in command)
    :param runtime:  runtime (job, nuclio, spark, dask ..) object/dict
                     store runtime specific details and preferences
    :param mode:     runtime mode, e.g. noctx, pass to bypass mlrun
    :param kfp:      reserved, flag indicating running within kubeflow pipeline

    :return: function object
    """
    kind, runtime = process_runtime(command, runtime, kind)
    command = get_in(runtime, 'spec.command', command)
    name = name or get_in(runtime, 'metadata.name', '')

    if not kind and not command:
        runner = HandlerRuntime()
    else:
        if kind in ['', 'local'] and command:
            runner = LocalRuntime.from_dict(runtime)
        elif kind in runtime_dict:
            runner = runtime_dict[kind].from_dict(runtime)
        else:
            raise Exception('unsupported runtime ({}) or missing command, '.format(kind)
                            + 'supported runtimes: {}'.format(
                              ','.join(list(runtime_dict.keys()) + ['local'])))

    if not name:
        # todo: regex check for valid name
        if command and kind not in ['remote']:
            name, _ = path.splitext(path.basename(command))
        else:
            name = 'mlrun-' + uuid.uuid4().hex[0:6]
    runner.metadata.name = name
    if project:
        runner.metadata.project = project
    if tag:
        runner.metadata.tag = tag
    if image:
        if kind in ['', 'handler', 'local']:
            raise ValueError('image should only be set with containerized '
                             'runtimes (job, mpijob, spark, ..), set kind=..')
        runner.spec.image = image
    if args:
        runner.spec.args = args
    runner.kfp = kfp
    if mode:
        runner.spec.mode = mode
    return runner


def process_runtime(command, runtime, kind):
    if runtime and hasattr(runtime, 'to_dict'):
        runtime = runtime.to_dict()
    if runtime and isinstance(runtime, dict):
        kind = kind or runtime.get('kind', '')
        command = command or get_in(runtime, 'spec.command', '')
    if '://' in command and command.startswith('http'):
        kind = kind or 'remote'
    if not runtime:
        runtime = {}
    update_in(runtime, 'spec.command', command)
    runtime['kind'] = kind
    if kind != 'remote':
        parse_command(runtime, command)
    else:
        update_in(runtime, 'spec.function_kind', 'mlrun')
    return kind, runtime


def parse_command(runtime, url):
    idx = url.find('#')
    if idx > -1:
        update_in(runtime, 'spec.image', url[:idx])
        url = url[idx+1:]

    if url:
        arg_list = url.split()
        update_in(runtime, 'spec.command', arg_list[0])
        update_in(runtime, 'spec.args', arg_list[1:])


def code_to_function(name: str = '', project: str = '', tag: str = '',
                     filename: str = '', handler='', runtime='', kind='',
                     image=None, code_output='', embed_code=True,
                     with_doc=False):
    """convert code or notebook to function object with embedded code
    code stored in the function spec and can be refreshed using .with_code()
    eliminate the need to build container images every time we edit the code

    :param name:       function name
    :param project:    function project (none for 'default')
    :param tag:        function tag (none for 'latest')
    :param filename:   blank for current notebook, or path to .py/.ipynb file
    :param handler:    name of function handler (if not main)
    :param runtime:    optional, runtime type local, job, dask, mpijob, ..
    :param kind:       optional, runtime type local, job, dask, mpijob, ..
    :param image:      optional, container image
    :param code_output: save the generated code (from notebook) in that path
    :param embed_code: embed the source code into the function spec

    :return:
           function object
    """
    filebase, _ = path.splitext(path.basename(filename))
    kind = runtime or kind  # for backwards computability

    def add_name(origin, name=''):
        name = filename or (name + '.ipynb')
        if not origin:
            return name
        return '{}:{}'.format(origin, name)

    if not embed_code and (not filename or filename.endswith('.ipynb')):
        raise ValueError('a valid code file must be specified '
                         'when not using the embed_code option')

    subkind = kind[kind.find(':') + 1:] if kind.startswith('nuclio:') else None
    code_origin = add_name(add_code_metadata(), name)

    name, spec, code = build_file(filename, name=name,
                                  handler=handler or 'handler',
                                  kind=subkind)
    spec_kind = get_in(spec, 'kind', '')
    if spec_kind not in ['', 'Function']:
        kind = spec_kind.lower()

        # if its a nuclio subkind, redo nb parsing
        if kind.startswith('nuclio:'):
            subkind = kind[kind.find(':') + 1:]
            name, spec, code = build_file(filename, name=name,
                                          handler=handler or 'handler',
                                          kind=subkind)

    if code_output:
        if filename == '' or filename.endswith('.ipynb'):
            with open(code_output, 'w') as fp:
                fp.write(code)
        else:
            raise ValueError('code_output option is only used with notebooks')

    if kind.startswith('nuclio'):
        r = RemoteRuntime()
        r.spec.function_kind = subkind
        if embed_code:
            update_in(spec, 'kind', 'Function')
            r.spec.base_spec = spec
            if with_doc:
                handlers = find_handlers(code)
                r.spec.entry_points = {h.name: as_func(h) for h in handlers}
        else:
            r.spec.source = filename
            r.spec.function_handler = handler

        if not name:
            raise ValueError('name must be specified')
        r.metadata.name = name
        r.metadata.project = project
        r.metadata.tag = tag
        r.spec.build.code_origin = code_origin
        return r

    if kind is None or kind in ['', 'Function']:
        raise ValueError('please specify the function kind')
    elif kind in ['local']:
        if not code_output and embed_code:
            raise ValueError('code_output path or embed_code=False should be'
                             ' specified for local runtime')
        r = LocalRuntime()
    elif kind in runtime_dict:
        r = runtime_dict[kind]()
    else:
        raise ValueError('unsupported runtime ({})'.format(kind))

    name, spec, code = build_file(filename, name=name)

    if not name:
        raise ValueError('name must be specified')
    h = get_in(spec, 'spec.handler', '').split(':')
    r.handler = h[0] if len(h) <= 1 else h[1]
    r.metadata = get_in(spec, 'spec.metadata')
    r.metadata.project = project
    r.metadata.name = name
    r.metadata.tag = tag
    r.spec.image = get_in(spec, 'spec.image', image)
    build = r.spec.build
    build.code_origin = code_origin
    build.base_image = get_in(spec, 'spec.build.baseImage')
    build.commands = get_in(spec, 'spec.build.commands')
    if code_output:
        r.spec.command = code_output
    elif not embed_code:
        r.spec.command = filename
    else:
        build.functionSourceCode = get_in(spec, 'spec.build.functionSourceCode')

    build.image = get_in(spec, 'spec.build.image')
    build.secret = get_in(spec, 'spec.build.secret')
    if r.kind != 'local':
        r.spec.env = get_in(spec, 'spec.env')
        for vol in get_in(spec, 'spec.volumes', []):
            r.spec.volumes.append(vol.get('volume'))
            r.spec.volume_mounts.append(vol.get('volumeMount'))

    if with_doc:
        handlers = find_handlers(code)
        r.spec.entry_points = {h[name]: h for h in handlers}
    return r


def as_func(handler):
    return FunctionEntrypoint(
        name=handler['name'],
        doc=handler['doc'],
        parameters=[as_entry(p) for p in handler['params']],
        outputs=[as_entry(p) for p in handler['returns']],
        lineno=handler['lineno'],
    )


def as_entry(param):
    return EntrypointParam(**param)
