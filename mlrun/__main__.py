#!/usr/bin/env python

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
import sys
from ast import literal_eval
from base64 import b64decode, b64encode
from os import environ, path
from pprint import pprint
from subprocess import Popen
from sys import executable

import click
import yaml

from tabulate import tabulate

from . import get_version
from .config import config as mlconf
from .builder import upload_tarball
from .datastore import get_object
from .db import get_run_db
from .k8s_utils import K8sHelper
from .model import RunTemplate
from .run import new_function, import_function_to_dict, import_function
from .runtimes import RemoteRuntime, RunError
from .utils import (list2dict, logger, run_keys, update_in, get_in,
                    parse_function_uri, dict_to_yaml)


@click.group()
def main():
    pass


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("url", type=str, required=False)
@click.option('--param', '-p', default='', multiple=True,
              help="parameter name and value tuples, e.g. -p x=37 -p y='text'")
@click.option('--inputs', '-i', multiple=True, help='input artifact')
@click.option('--outputs', '-o', multiple=True, help='output artifact/result for kfp')
@click.option('--in-path', help='default input path/url (prefix) for artifact')
@click.option('--out-path', help='default output path/url (prefix) for artifact')
@click.option('--secrets', '-s', multiple=True, help='secrets file=<filename> or env=ENV_KEY1,..')
@click.option('--uid', help='unique run ID')
@click.option('--name', help='run name')
@click.option('--workflow', help='workflow name/id')
@click.option('--project', help='project name/id')
@click.option('--db', default='', help='save run results to path or DB url')
@click.option('--runtime', '-r', default='', help='function spec dict, for pipeline usage')
@click.option('--kfp', is_flag=True, help='running inside Kubeflow Piplines, do not use')
@click.option('--hyperparam', '-x', default='', multiple=True,
              help='hyper parameters (will expand to multiple tasks) e.g. --hyperparam p2=[1,2,3]')
@click.option('--param-file', default='', help='path to csv table of execution (hyper) params')
@click.option('--selector', default='', help='how to select the best result from a list, e.g. max.accuracy')
@click.option('--func-url', '-f', default='', help='path/url of function yaml or function '
                                                   'yaml or db://<project>/<name>[:tag]')
@click.option('--task', default='', help='path/url to task yaml')
@click.option('--handler', default='', help='invoke function handler inside the code file')
@click.option('--mode', help='special run mode noctx | pass')
@click.option('--schedule', help='cron schedule')
@click.option('--from-env', is_flag=True, help='read the spec from the env var')
@click.option('--dump', is_flag=True, help='dump run results as YAML')
@click.option('--image', default='', help='container image')
@click.option('--watch', '-w', is_flag=True, help='watch/tail run log')
@click.argument('run_args', nargs=-1, type=click.UNPROCESSED)
def run(url, param, inputs, outputs, in_path, out_path, secrets, uid,
        name, workflow, project, db, runtime, kfp, hyperparam, param_file,
        selector, func_url, task, handler, mode, schedule, from_env, dump,
        image, watch, run_args):
    """Execute a task and inject parameters."""

    config = environ.get('MLRUN_EXEC_CONFIG')
    if from_env and config:
        config = json.loads(config)
        runobj = RunTemplate.from_dict(config)
    elif task:
        obj = get_object(task)
        task = yaml.load(obj, Loader=yaml.FullLoader)
        runobj = RunTemplate.from_dict(task)
    else:
        runobj = RunTemplate()

    code = environ.get('MLRUN_EXEC_CODE')
    if from_env and code:
        code = b64decode(code).decode('utf-8')
        with open('main.py', 'w') as fp:
            fp.write(code)
        url = url or 'main.py'

    set_item(runobj.metadata, uid, 'uid')
    set_item(runobj.metadata, name, 'name')
    set_item(runobj.metadata, project, 'project')

    if workflow:
        runobj.metadata.labels['workflow'] = workflow

    if db:
        mlconf.dbpath = db

    if func_url:
        if func_url.startswith('db://'):
            func_url = func_url[5:]
            project, name, tag = parse_function_uri(func_url)
            mldb = get_run_db(mlconf.dbpath).connect()
            runtime = mldb.get_function(name, project, tag)
        else:
            func_url = 'function.yaml' if func_url == '.' else func_url
            runtime = import_function_to_dict(func_url, {})
        kind = get_in(runtime, 'kind', '')
        if kind not in ['', 'local'] and url:
            if path.isfile(url) and url.endswith('.py'):
                with open(url) as fp:
                    body = fp.read()
                based = b64encode(body.encode('utf-8')).decode('utf-8')
                logger.info('packing code at {}'.format(url))
                update_in(runtime, 'spec.build.functionSourceCode', based)
                url = ''
                update_in(runtime, 'spec.command', '')

    elif runtime:
        runtime = py_eval(runtime)
        if not isinstance(runtime, dict):
            print('runtime parameter must be a dict, not {}'.format(type(runtime)))
            exit(1)
        if kfp:
            print('Runtime:')
            pprint(runtime)
            print('Run:')
            pprint(runobj.to_dict())
    else:
        runtime = {}
    if url:
        update_in(runtime, 'spec.command', url)
    if run_args:
        update_in(runtime, 'spec.args', list(run_args))
    if image:
        update_in(runtime, 'spec.image', image)
    set_item(runobj.spec, handler, 'handler')
    set_item(runobj.spec, param, 'parameters', fill_params(param))
    set_item(runobj.spec, hyperparam, 'hyperparams', fill_params(hyperparam))
    set_item(runobj.spec, param_file, 'param_file')
    set_item(runobj.spec, selector, 'selector')

    set_item(runobj.spec, inputs, run_keys.inputs, list2dict(inputs))
    set_item(runobj.spec, in_path, run_keys.input_path)
    set_item(runobj.spec, out_path, run_keys.output_path)
    set_item(runobj.spec, outputs, run_keys.outputs, list(outputs))
    set_item(runobj.spec, secrets, run_keys.secrets, line2keylist(secrets, 'kind', 'source'))
    try:
        update_in(runtime, 'metadata.name', name, replace=False)
        fn = new_function(runtime=runtime, kfp=kfp, mode=mode)
        fn.is_child = from_env and not kfp
        resp = fn.run(runobj, watch=watch, schedule=schedule)
        if resp and dump:
            print(resp.to_yaml())
    except RunError as err:
        print('runtime error: {}'.format(err))
        exit(1)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("func_url", type=str, required=False)
@click.option('--name', help='function name')
@click.option('--project', help='project name')
@click.option('--tag', default='', help='function tag')
@click.option('--image', '-i', help='target image path')
@click.option('--source', '-s', default='',
              help='location/url of the source files dir/tar')
@click.option('--base-image', '-b', help='base docker image')
@click.option('--command', '-c', default='', multiple=True,
              help="build commands, e.g. '-c pip install pandas'")
@click.option('--secret-name', default='', help='container registry secret name')
@click.option('--archive', '-a', default='', help='destination archive for code (tar)')
@click.option('--silent', is_flag=True, help='do not show build logs')
@click.option('--with-mlrun', is_flag=True, help='add MLRun package')
@click.option('--db', default='', help='save run results to path or DB url')
@click.option('--runtime', '-r', default='', help='function spec dict, for pipeline usage')
@click.option('--kfp', is_flag=True, help='running inside Kubeflow Piplines, do not use')
def build(func_url, name, project, tag, image, source, base_image, command,
          secret_name, archive, silent, with_mlrun, db, runtime, kfp):
    """Build a container image from code and requirements."""

    if runtime:
        runtime = py_eval(runtime)
        if not isinstance(runtime, dict):
            print('runtime parameter must be a dict, not {}'.format(type(runtime)))
            exit(1)
        if kfp:
            print('Runtime:')
            pprint(runtime)
        func = new_function(runtime=runtime)
    elif func_url.startswith('db://'):
        func_url = func_url[5:]
        project, name, tag = parse_function_uri(func_url)
        func = import_function(func_url, db=db)
    elif func_url:
        func_url = 'function.yaml' if func_url == '.' else func_url
        func = import_function(func_url, db=db)
    else:
        print('please specify the function path or url')
        exit(1)

    meta = func.metadata
    meta.project = project or meta.project or mlconf.default_project
    meta.name = name or meta.name
    meta.tag = tag or meta.tag

    b = func.spec.build
    if func.kind not in ['', 'local']:
        b.base_image = base_image or b.base_image
        b.commands = list(command) or b.commands
        b.image = image or b.image
        b.secret = secret_name or b.secret

    if source.endswith('.py'):
        if not path.isfile(source):
            print('source file doesnt exist ({})'.format(source))
            exit(1)
        with open(source) as fp:
            body = fp.read()
        based = b64encode(body.encode('utf-8')).decode('utf-8')
        logger.info('packing code at {}'.format(source))
        b.functionSourceCode = based
        func.spec.command = ''
    else:
        b.source = source or b.source
        # todo: upload stuff

    archive = archive or mlconf.default_archive
    if archive:
        src = b.source or './'
        logger.info('uploading data from {} to {}'.format(src, archive))
        target = archive if archive.endswith('/') else archive + '/'
        target += 'src-{}-{}-{}.tar.gz'.format(meta.project, meta.name,
                                               meta.tag or 'latest')
        upload_tarball(src, target)
        # todo: replace function.yaml inside the tar
        b.source = target

    if hasattr(func, 'deploy'):
        logger.info('remote deployment started')
        try:
            func.deploy(with_mlrun=with_mlrun, watch=not silent, is_kfp=kfp)
        except Exception as err:
            print('deploy error, {}'.format(err))
            exit(1)

        if kfp:
            state = func.status.state
            image = func.spec.image
            print('function built, state={} image={}'.format(state, image))
            with open('/tmp/state', 'w') as fp:
                fp.write(state)
            with open('/tmp/image', 'w') as fp:
                fp.write(func.full_image_path(image))
    else:
        print('function does not have a deploy() method')
        exit(1)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("spec", type=str, required=False)
@click.option('--source', '-s', default='', help='location/url of the source')
@click.option('--dashboard', '-d', default='', help='nuclio dashboard url')
@click.option('--project', '-p', default='', help='container registry secret name')
@click.option('--model', '-m', multiple=True, help='input artifact')
@click.option('--kind', '-k', default=None, help='runtime sub kind')
@click.option('--tag', default='', help='version tag')
@click.option('--env', '-e', multiple=True, help='environment variables')
@click.option('--verbose', is_flag=True, help='verbose log')
def deploy(spec, source, dashboard, project, model, tag, kind, env, verbose):
    """Deploy model or function"""
    if spec:
        runtime = py_eval(spec)
    else:
        runtime = {}
    if not isinstance(runtime, dict):
        print('runtime parameter must be a dict, not {}'.format(type(runtime)))
        exit(1)

    f = RemoteRuntime.from_dict(runtime)
    f.spec.source = source
    if kind:
        f.spec.function_kind = kind
    if env:
        for k, v in list2dict(env).items():
            f.set_env(k, v)
    f.verbose = verbose
    if model:
        models = list2dict(model)
        for k, v in models.items():
            f.add_model(k, v)

    try:
        addr = f.deploy(dashboard=dashboard, project=project, tag=tag, kind=kind)
    except Exception as err:
        print('deploy error: {}'.format(err))
        exit(1)

    print('function deployed, address={}'.format(addr))
    with open('/tmp/output', 'w') as fp:
        fp.write(addr)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("pod", type=str)
@click.option('--namespace', '-n', help='kubernetes namespace')
@click.option('--timeout', '-t', default=600, show_default=True,
              help='timeout in seconds')
def watch(pod, namespace, timeout):
    """Read current or previous task (pod) logs."""
    k8s = K8sHelper(namespace)
    status = k8s.watch(pod, namespace, timeout)
    print('Pod {} last status is: {}'.format(pod, status))


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('kind', type=str)
@click.argument('name', type=str, default='', required=False)
@click.option('--selector', '-s', default='', help='label selector')
@click.option('--namespace', '-n', help='kubernetes namespace')
@click.option('--uid', help='unique ID')
@click.option('--project', help='project name')
@click.option('--tag', '-t', default='', help='artifact/function tag')
@click.option('--db', help='db path/url')
@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def get(kind, name, selector, namespace, uid, project, tag, db, extra_args):
    """List/get one or more object per kind/class."""
    if kind.startswith('po'):
        k8s = K8sHelper(namespace)
        if name:
            resp = k8s.get_pod(name, namespace)
            print(resp)
            return

        items = k8s.list_pods(namespace, selector)
        print('{:10} {:16} {:8} {}'.format('state', 'started', 'type', 'name'))
        for i in items:
            task = i.metadata.labels.get('mlrun/class', '')
            if task:
                name = i.metadata.name
                state = i.status.phase
                start = ''
                if i.status.start_time:
                    start = i.status.start_time.strftime("%b %d %H:%M:%S")
                print('{:10} {:16} {:8} {}'.format(state, start, task, name))
    elif kind.startswith('run'):
        mldb = get_run_db(db or mlconf.dbpath).connect()
        if name:
            run = mldb.read_run(name, project=project)
            print(dict_to_yaml(run))
            return

        runs = mldb.list_runs(uid=uid, project=project)
        df = runs.to_df()[['name', 'uid', 'iter', 'start', 'state', 'parameters', 'results']]
        #df['uid'] = df['uid'].apply(lambda x: '..{}'.format(x[-6:]))
        df['start'] = df['start'].apply(time_str)
        df['parameters'] = df['parameters'].apply(dict_to_str)
        df['results'] = df['results'].apply(dict_to_str)
        print(tabulate(df, headers='keys'))

    elif kind.startswith('art'):
        mldb = get_run_db(db or mlconf.dbpath).connect()
        artifacts = mldb.list_artifacts(name, project=project, tag=tag)
        df = artifacts.to_df()[['tree', 'key', 'iter', 'kind', 'path', 'hash', 'updated']]
        df['tree'] = df['tree'].apply(lambda x: '..{}'.format(x[-8:]))
        df['hash'] = df['hash'].apply(lambda x: '..{}'.format(x[-6:]))
        print(tabulate(df, headers='keys'))

    elif kind.startswith('func'):
        mldb = get_run_db(db or mlconf.dbpath).connect()
        if name:
            f = mldb.get_function(name, project=project, tag=tag)
            print(dict_to_yaml(f))
            return

        functions = mldb.list_functions(name, project=project)
        lines = []
        headers = ['kind', 'state', 'name:tag', 'hash']
        for f in functions:
            line = [
                get_in(f, 'kind', ''),
                get_in(f, 'status.state', ''),
                '{}:{}'.format(get_in(f, 'metadata.name'), get_in(f, 'metadata.tag', '')),
                get_in(f, 'metadata.hash', ''),
            ]
            lines.append(line)
        print(tabulate(lines, headers=headers))
    else:
        print('currently only get pods | runs | artifacts | func [name] are supported')


@main.command()
@click.option('--port', '-p', help='port to listen on', type=int)
@click.option('--dirpath', '-d', help='database directory (dirpath)')
def db(port, dirpath):
    """Run HTTP api/database server"""
    env = environ.copy()
    if port is not None:
        env['MLRUN_httpdb__port'] = str(port)
    if dirpath is not None:
        env['MLRUN_httpdb__dirpath'] = dirpath

    cmd = [executable, '-m', 'mlrun.db.httpd']
    child = Popen(cmd, env=env)
    returncode = child.wait()
    if returncode != 0:
        raise SystemExit(returncode)


@main.command()
def version():
    """get mlrun version"""
    print('MLRun version: {}'.format(get_version()))


@main.command()
@click.argument('uid', type=str)
@click.option('--project', '-p', help='project name')
@click.option('--offset', type=int, default=0, help='byte offset')
@click.option('--db', help='api and db service path/url')
@click.option('--watch', '-w', is_flag=True, help='watch/follow log')
def logs(uid, project, offset, db, watch):
    """Get or watch task logs"""
    mldb = get_run_db(db or mlconf.dbpath).connect()
    if mldb.kind == 'http':
        state = mldb.watch_log(uid, project, watch=watch, offset=offset)
    else:
        state, text = mldb.get_log(uid, project, offset=offset)
        if text:
            print(text.decode())

    if state:
        print('final state: {}'.format(state))


@main.command()
@click.option('--api', help='api and db service path/url')
@click.option('--namespace', '-n', help='kubernetes namespace')
@click.option('--pending', '-p', is_flag=True,
              help='clean pending pods as well')
@click.option('--running', '-r', is_flag=True,
              help='clean running pods as well')
def clean(api, namespace, pending, running):
    """Clean completed or failed pods/jobs"""
    k8s = K8sHelper(namespace)
    #mldb = get_run_db(db or mlconf.dbpath).connect()
    items = k8s.list_pods(namespace)
    states = ['Succeeded', 'Failed']
    if pending:
        states.append('Pending')
    if running:
        states.append('Running')
    print('{:10} {:16} {:8} {}'.format('state', 'started', 'type', 'name'))
    for i in items:
        task = i.metadata.labels.get('mlrun/class', '')
        state = i.status.phase
        # todo: clean mpi, spark, .. jobs (+CRDs)
        if task and task in ['build', 'job', 'dask'] and state in states:
            name = i.metadata.name
            start = ''
            if i.status.start_time:
                start = i.status.start_time.strftime("%b %d %H:%M:%S")
            print('{:10} {:16} {:8} {}'.format(state, start, task, name))
            k8s.del_pod(name)


@main.command(name='config')
def show_config():
    """Show configuration & exit"""
    print(mlconf.dump_yaml())


def fill_params(params):
    params_dict = {}
    for param in params:
        i = param.find('=')
        if i == -1:
            continue
        key, value = param[:i].strip(), param[i + 1:].strip()
        if key is None:
            raise ValueError(
                'cannot find param key in line ({})'.format(param))
        params_dict[key] = py_eval(value)
    return params_dict


def py_eval(data):
    try:
        value = literal_eval(data)
        return value
    except (SyntaxError, ValueError):
        return data


def set_item(obj, item, key, value=None):
    if item:
        if value:
            setattr(obj, key, value)
        else:
            setattr(obj, key, item)


def line2keylist(lines: list, keyname='key', valname='path'):
    out = []
    for line in lines:
        i = line.find('=')
        if i == -1:
            raise ValueError('cannot find "=" in line ({}={})'.format(keyname, valname))
        key, value = line[:i].strip(), line[i + 1:].strip()
        if key is None:
            raise ValueError('cannot find key in line ({}={})'.format(keyname, valname))
        value = path.expandvars(value)
        out += [{keyname: key, valname: value}]
    return out


def time_str(x):
    try:
        return x.strftime("%b %d %H:%M:%S")
    except ValueError:
        return ''


def dict_to_str(struct: dict):
    if not struct:
        return []
    return ','.join(['{}={}'.format(k, v) for k, v in struct.items()])


if __name__ == "__main__":
    main()
