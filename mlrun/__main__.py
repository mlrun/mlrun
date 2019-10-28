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
from ast import literal_eval
from base64 import b64decode
from os import environ, path
from pprint import pprint
from subprocess import Popen
from sys import executable

import click

from tabulate import tabulate

from .config import config as mlconf
from .builder import build_image
from .db import get_run_db
from .k8s_utils import k8s_helper
from .model import RunTemplate
from .run import new_function, import_function_to_dict
from .runtimes import RemoteRuntime, RunError
from .utils import list2dict, logger, run_keys, update_in


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
@click.option('--rundb', default='', help='save run results to path or DB url')
@click.option('--runtime', '-r', default='', help='runtime environment e.g. local, remote, nuclio, mpi')
@click.option('--kfp', is_flag=True, help='running inside Kubeflow Piplines')
@click.option('--hyperparam', '-x', default='', multiple=True,
              help='hyper parameters (will expand to multiple tasks) e.g. --hyperparam p2=[1,2,3]')
@click.option('--param-file', default='', help='path to csv table of execution (hyper) params')
@click.option('--selector', default='', help='how to select the best result from a list, e.g. max.accuracy')
@click.option('--func-url', '-f', default='', help='path/url of function yaml')
@click.option('--handler', default='', help='invoke function handler inside the code file')
@click.option('--mode', default='', help='run mode e.g. noctx')
@click.option('--from-env', is_flag=True, help='read the spec from the env var')
@click.option('--dump', is_flag=True, help='dump run results as YAML')
@click.argument('run_args', nargs=-1, type=click.UNPROCESSED)
def run(url, param, inputs, outputs, in_path, out_path, secrets, uid,
        name, workflow, project, rundb, runtime, kfp, hyperparam, param_file,
        selector, func_url, handler, mode, from_env, dump, run_args):
    """Execute a task and inject parameters."""

    config = environ.get('MLRUN_EXEC_CONFIG')
    if from_env and config:
        config = json.loads(config)
        runobj = RunTemplate.from_dict(config)
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

    if rundb:
        mlconf.dbpath = rundb

    if func_url:
        runtime = import_function_to_dict(func_url, {})
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
        resp = new_function(runtime=runtime, kfp=kfp, mode=mode).run(runobj)
        if resp and dump:
            print(resp.to_yaml())
    except RunError as err:
        print('runtime error: {}'.format(err))
        exit(1)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("dest", type=str)
@click.option('--command', '-c', default='', multiple=True,
              help="build commands, e.g. '-p pip install pandas'")
@click.option('--source', '-s', help='location/url of the source files dir/tar')
@click.option('--base-image', '-b', help='base docker image')
@click.option('--secret-name', default='my-docker', help='container registry secret name')
@click.option('--requirements', '-r', help='python package requirements file path')
@click.option('--namespace', help='kubernetes namespace')
@click.option('--silent', is_flag=True, help='do not show build logs')
@click.option('--inline', '-i', is_flag=True, help='inline code (for single file)')
def build(dest, command, source, base_image, secret_name,
          requirements, namespace, silent, inline):
    """Build a container image from code and requirements."""

    inline_code = None
    cmd = list(command)
    if inline:
        with open(source, 'r') as fp:
            inline_code = fp.read()
        source = None
        if requirements:
            with open(requirements, 'r') as fp:
                requirements = fp.readlines()

    print(dest, cmd, source, inline_code, base_image,
          secret_name, requirements, namespace)

    status = build_image(dest, command, source,
                         inline_code=inline_code,
                         base_image=base_image,
                         secret_name=secret_name,
                         requirements=requirements,
                         namespace=namespace,
                         interactive=not silent)

    logger.info('build completed with {}'.format(status))
    if status in ['failed', 'error']:
        exit(1)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("spec", type=str)
@click.option('--source', '-s', default='', help='location/url of the source')
@click.option('--dashboard', '-d', default='', help='nuclio dashboard url')
@click.option('--project', '-p', default='', help='container registry secret name')
@click.option('--model', '-m', multiple=True, help='input artifact')
@click.option('--kind', '-k', default='nuclio', help='runtime kind')
@click.option('--tag', default='', help='version tag')
@click.option('--verbose', is_flag=True, help='verbose log')
def deploy(spec, source, dashboard, project, model, tag, kind, verbose):
    """Deploy model"""
    runtime = py_eval(spec)
    if not isinstance(runtime, dict):
        print('runtime parameter must be a dict, not {}'.format(type(runtime)))
        exit(1)

    f = RemoteRuntime.from_dict(runtime)
    f.verbose = verbose
    if model:
        models = list2dict(model)
        for k, v in models.items():
            f.add_model(k, v)

    addr = f.deploy(source=source, dashboard=dashboard, project=project, tag=tag)
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
    k8s = k8s_helper(namespace)
    status = k8s.watch(pod, namespace, timeout)
    print('Pod {} last status is: {}'.format(pod, status))


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('kind', type=str)
@click.argument('name', type=str, default='', required=False)
@click.option('--selector', '-s', default='', help='label selector')
@click.option('--namespace', '-n', help='kubernetes namespace')
@click.option('--uid', help='unique ID')
@click.option('--project', help='project name')
@click.option('--tag', default='', help='artifact tag')
@click.option('--db', help='db path/url')
@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def get(kind, name, selector, namespace, uid, project, tag, db, extra_args):
    """List/get one or more object per kind/class."""
    if kind.startswith('po'):
        k8s = k8s_helper(namespace)
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
                start = i.status.start_time.strftime("%b %d %H:%M:%S")
                print('{:10} {:16} {:8} {}'.format(state, start, task, name))
    elif kind.startswith('run'):
        mldb = get_run_db(db).connect()
        runs = mldb.list_runs(name, uid=uid, project=project)
        df = runs.to_df()[['name', 'uid', 'iter', 'start', 'state', 'parameters', 'results']]
        #df['uid'] = df['uid'].apply(lambda x: '..{}'.format(x[-6:]))
        df['start'] = df['start'].apply(time_str)
        df['parameters'] = df['parameters'].apply(dict_to_str)
        df['results'] = df['results'].apply(dict_to_str)
        print(tabulate(df, headers='keys'))

    elif kind.startswith('art'):
        mldb = get_run_db(db).connect()
        artifacts = mldb.list_artifacts(name, project=project, tag=tag)
        df = artifacts.to_df()[['tree', 'key', 'iter', 'kind', 'path', 'hash', 'updated']]
        df['tree'] = df['tree'].apply(lambda x: '..{}'.format(x[-8:]))
        df['hash'] = df['hash'].apply(lambda x: '..{}'.format(x[-6:]))
        # df['start'] = df['start'].apply(time_str)
        # df['parameters'] = df['parameters'].apply(dict_to_str)
        # df['results'] = df['results'].apply(dict_to_str)
        print(tabulate(df, headers='keys'))

    else:
        print('currently only get pods [name] is supported')


@main.command()
@click.option('--port', '-p', help='port to listen on', type=int)
@click.option('--dirpath', '-d', help='database directory (dirpath)')
def db(port, dirpath):
    """Run HTTP database server"""
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
