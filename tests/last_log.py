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

"""Print last http test log"""

from pathlib import Path


def mtime(path: Path):
    return path.stat().st_mtime


if __name__ == '__main__':
    test_dir = sorted(Path('/tmp').glob('mlrun-test*'), key=mtime)[-1]
    log_file = test_dir / 'httpd.log'
    with log_file.open() as fp:
        print(fp.read())
    print(f'\n\n{test_dir}')
