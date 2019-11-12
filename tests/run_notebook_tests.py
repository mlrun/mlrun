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

from pathlib import Path
from shutil import get_terminal_size
from subprocess import run
import traceback


here = Path(__file__).absolute().parent
notebooks_dir = here / 'notebooks'
tmp_dockerfile = Path('/tmp/Dockerfile.mlrul-test-nb')
tag_name = 'mlrun/test-notebook'


def test_notebook(notebook, template):
    code = template.format(notebook=notebook)
    with tmp_dockerfile.open('w') as out:
        out.write(code)

    cmd = [
        'docker', 'build',
        '--tag', tag_name,
        '--file', tmp_dockerfile,
        '.',
    ]
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
            test_notebook(name, template)
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
