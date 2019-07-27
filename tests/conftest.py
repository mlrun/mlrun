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

import pytest
import pathlib
from os.path import abspath, dirname
from os import environ

here = dirname(abspath(__file__))
results = f'{here}/results'

rundb_path = f'{results}/rundb'
out_path = f'{results}/out'

pathlib.Path(f'{results}/kfp').mkdir(parents=True, exist_ok=True)
environ['KFPMETA_OUT_DIR'] = f'{results}/kfp/'

