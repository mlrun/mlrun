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
from base64 import b64decode
from os import path, environ
import click
from ast import literal_eval

from .k8s_utils import k8s_helper
from .run import run_start
from .runtimes import RunError
from .utils import run_keys, dict_to_yaml, logger
from .builder import build_image
from .model import RunTemplate


@click.group()
def main():
    pass

@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("url", type=str)
@click.option('--param', '-p', default='', multiple=True,
              help="parameter name and value tuples, e.g. -p x=37 -p y='text'")
@click.option('--in-artifact', '-i', multiple=True, help='input artifact')
@click.option('--out-artifact', '-o', multiple=True, help='output artifact')
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
@click.option('--mode', default='', help='run mode e.g. noctx')
@click.option('--from-env', is_flag=True, help='read the spec from the env var')
@click.argument('run_args', nargs=-1, type=click.UNPROCESSED)
def run(url, param, in_artifact, out_artifact, in_path, out_path, secrets,
        uid, name, workflow, project, rundb, runtime, kfp, hyperparam,
        param_file, mode, from_env, run_args):
    """Execute a task and inject parameters."""

    config = environ.get('MLRUN_EXEC_CONFIG')
    if from_env and config:
        config = py_eval(config)
        if not isinstance(config, dict):
            print(f'config env var must be a dict')
            exit(1)
        runobj = RunTemplate.from_dict(config)
    else:
        runobj = RunTemplate()

    code = environ.get('MLRUN_EXEC_CODE')
    if from_env and code:
        code = b64decode(code).decode('utf-8')
        with open('main.py', 'w') as fp:
            fp.write(code)

    set_item(runobj.metadata, uid, 'uid')
    set_item(runobj.metadata, name, 'name')
    set_item(runobj.metadata, project, 'project')

    if workflow:
        runobj.metadata.labels['workflow'] = workflow

    if runtime:
        runtime = py_eval(runtime)
        if not isinstance(runtime, dict):
            print(f'runtime parameter must be a dict')
            exit(1)
    else:
        runtime = {}

    if url:
        runtime['command'] = url
    if run_args:
        runtime['args'] = list(run_args)
    set_item(runobj.spec, param, 'parameters', fill_params(param))
    set_item(runobj.spec, hyperparam, 'hyperparams', fill_params(hyperparam))
    set_item(runobj.spec, param_file, 'param_file')

    set_item(runobj.spec, in_artifact, run_keys.input_objects, line2keylist(in_artifact))
    set_item(runobj.spec, in_path, run_keys.input_path)
    set_item(runobj.spec, out_path, run_keys.output_path)
    set_item(runobj.spec, out_artifact, run_keys.output_artifacts, line2keylist(out_artifact))
    set_item(runobj.spec, secrets, run_keys.secrets, line2keylist(secrets, 'kind', 'source'))
    try:
        resp = run_start(runobj, runtime=runtime, rundb=rundb, kfp=kfp, mode=mode)
        if resp:
            print(dict_to_yaml(resp))
    except RunError as err:
        print(f'runtime error: {err}')
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

    build_image(dest, command, source,
                inline_code=inline_code,
                base_image=base_image,
                secret_name=secret_name,
                requirements=requirements,
                namespace=namespace,
                interactive=not silent)


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("pod", type=str)
@click.option('--namespace', '-n', help='kubernetes namespace')
@click.option('--timeout', '-t', default=600, show_default=True,
              help='timeout in seconds')
def watch(pod, namespace, timeout):
    """read current or previous task (pod) logs."""
    k8s = k8s_helper(namespace or 'default-tenant')
    status = k8s.watch(pod, namespace, timeout)
    print('Pod {} last status is: {}'.format(pod, status))


@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('kind', type=str)
@click.argument('name', type=str, required=False)
@click.option('--selector', '-s', default='', help='label selector')
@click.option('--namespace', '-n', help='kubernetes namespace')
@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def get(kind, name, selector, namespace, extra_args):
    """List/get one or more object per kind/class."""
    if kind.startswith('po'):
        k8s = k8s_helper(namespace or 'default-tenant')
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
    else:
        print('currently only ls pods is supported')


def fill_params(param):
    params_dict = {}
    for param in param:
        i = param.find('=')
        if i == -1:
            continue
        key, value = param[:i].strip(), param[i + 1:].strip()
        if key is None:
            raise ValueError(f'cannot find param key in line ({param})')
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


if __name__ == "__main__":
    main()


