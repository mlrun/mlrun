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

from sys import executable, stderr
from subprocess import run, PIPE
from conftest import rundb_path, out_path, tag_test, here, examples_path, root_path


def exec_main(op, args):
    cmd = [executable, '-m', 'mlrun', op]
    if args:
        cmd += args
    out = run(cmd, stdout=PIPE, stderr=PIPE, cwd=root_path)
    if out.returncode != 0:
        print(out.stderr.decode('utf-8'), file=stderr)
        raise Exception(out.stderr.decode('utf-8'))
    return out.stdout.decode('utf-8')


def exec_run(cmd, args, test):
    args = args + ['--name', test, '--dump', cmd]
    return exec_main('run', args)


def param_list(params: dict, flag='-p'):
    l = []
    for k, v in params.items():
        l += [flag, f'{k}={v}']
    return l


def test_main_run_basic():
    out = exec_run(f'{examples_path}/training.py',
                   param_list(dict(p1=5, p2='"aaa"')),
                   'test_main_run_basic')
    print(out)
    assert out.find('state: completed') != -1, out


def test_main_run_hyper():
    out = exec_run(f'{examples_path}/training.py',
                   param_list(dict(p2=[4, 5, 6]), '-x'),
                   'test_main_run_hyper')
    print(out)
    assert out.find('state: completed') != -1, out
    assert out.find('iterations:') != -1, out


def test_main_run_noctx():
    out = exec_run(f'{here}/no_ctx.py',
                   ['--mode', 'noctx'] + param_list(dict(p1=5, p2='"aaa"')),
                   'test_main_run_noctx')
    print(out)
    assert out.find('state: completed') != -1, out
