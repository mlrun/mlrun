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

from os import environ
from pathlib import Path
from subprocess import run

import pytest

from conftest import here, notebooks_dir

nb_test = 'MLRUN_NB_TEST' in environ

with (here / 'Dockerfile.test-nb').open() as fp:
    dockerfile_template = fp.read()

tmp_dockerfile = Path('/tmp/Dockerfile.mlrul-test-nb')
tag_name = 'mlrun/test-notebook'

test_notebooks = (nb.name for nb in notebooks_dir.glob('*.ipynb'))


@pytest.mark.skipif(not nb_test, reason='no notebook testing')
@pytest.mark.parametrize('notebook', test_notebooks)
def test_notebook(notebook):
    code = dockerfile_template.format(notebook=notebook)
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
