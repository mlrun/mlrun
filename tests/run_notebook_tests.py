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

# This is ran by CI system. Use standard library functions only

import traceback
from os import environ
from pathlib import Path
from shutil import get_terminal_size
from subprocess import run

here = Path(__file__).absolute().parent
notebooks_dir = here / 'notebooks'
tmp_dockerfile = Path('./Dockerfile.mlrun-test-nb')


def args_from_env():
    args, cmd = [], []
    for name in environ:
        if not name.startswith('MLRUN_'):
            continue
        value = environ[name]
        args.append(f'ARG {name}')
        cmd.extend(['--build-arg', f'{name}={value}'])

    args = '\n'.join(args)
    return args, cmd


# Must not start with "test_", otherwise pytset will catch it
def check_notebook(notebook, template):
    args, args_cmd = args_from_env()
    code = template.format(notebook=notebook, args=args)
    with tmp_dockerfile.open('w') as out:
        out.write(code)

    cmd = ['docker', 'build', '--file', str(tmp_dockerfile)] + args_cmd + ['.']
    out = run(cmd)
    assert out.returncode == 0, 'cannot build'


if __name__ == '__main__':
    from argparse import ArgumentParser, FileType
    parser = ArgumentParser(description='run notebook tests')
    parser.add_argument(
        'notebooks', metavar='NOTEBOOK', nargs='*', help='notebook to test',
        type=FileType('r'))
    args = parser.parse_args()

    notebooks = list(args.notebooks or notebooks_dir.glob('*.ipynb'))

    with (here / 'Dockerfile.test-nb').open() as fp:
        template = fp.read()

    count = len(notebooks)
    failed = []
    for i, nb in enumerate(notebooks, 1):
        name = nb.name
        print(f'[NB] Testing {name!r} ({i}/{count})')
        tb = None
        result = 'OK'
        try:
            check_notebook(name, template)
        except Exception as err:
            result = f'FAIL ({err})'
            failed.append(name)
            tb = traceback.format_exc()
        print(f'[NB] {name!r} {result}')
        if tb:
            print(tb)

    if failed:
        width = get_terminal_size().columns
        print('\n' + ('-' * width))
        print('[NB] failed: ' + ', '.join(sorted(failed)))
        raise SystemExit(1)
