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

from io import BytesIO
import pandas as pd
import sys
from copy import deepcopy
from ..model import RunObject
from ..utils import get_in, logger


def get_generator(spec, execution):
    if spec.hyperparams:
        return GridGenerator(spec.hyperparams)
    elif spec.param_file:
        obj = execution.get_input('param_file.csv', spec.param_file)
        return ListGenerator(obj.get())


class TaskGenerator:
    def generate(self, run: RunObject):
        pass


class GridGenerator(TaskGenerator):
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def generate(self, run: RunObject):
        i = 0
        params = self.grid_to_list()
        max = len(next(iter(params.values())))

        while i < max:
            newrun = deepcopy(run)
            newrun.spec.hyperparams = None
            param_dict = newrun.spec.parameters or {}
            for key, values in params.items():
                param_dict[key] = values[i]
            newrun.spec.parameters = param_dict
            newrun.metadata.iteration = i + 1
            i += 1
            yield newrun

    def grid_to_list(self):
        arr = {}
        lastlen = 1
        for pk, pv in self.hyperparams.items():
            for p in arr.keys():
                arr[p] = arr[p] * len(pv)
            expanded = []
            for i in range(len(pv)):
                expanded += [pv[i]] * lastlen
            arr[pk] = expanded
            lastlen = lastlen * len(pv)

        return arr


class ListGenerator(TaskGenerator):
    def __init__(self, body):

        self.df = pd.read_csv(BytesIO(body), encoding='utf-8')

    def generate(self, run: RunObject):
        i = 0
        for _, row in self.df.iterrows():
            newrun = deepcopy(run)
            newrun.spec.param_file = None
            param_dict = newrun.spec.parameters or {}
            for key, values in row.to_dict().items():
                param_dict[key] = values
            newrun.spec.parameters = param_dict
            newrun.metadata.iteration = i + 1
            i += 1
            yield newrun


def selector(results: list, criteria):
    if not criteria:
        return 0, 0

    idx = criteria.find('.')
    if idx < 0:
        op = 'max'
    else:
        op = criteria[:idx]
        criteria = criteria[idx + 1:]

    best_id = 0
    best_item = 0
    if op == 'max':
        best_val = sys.float_info.min
    elif op == 'min':
        best_val = sys.float_info.max
    else:
        logger.error('unsupported selector {}.{}'.format(op, criteria))
        return 0, 0

    i = 0
    for task in results:
        state = get_in(task, ['status', 'state'])
        id = get_in(task, ['metadata', 'iteration'])
        val = get_in(task, ['status', 'results', criteria])
        if isinstance(val, str):
            try:
                val = float(val)
            except Exception:
                val = None
        if state != 'error' and val is not None:
            if (op == 'max' and val > best_val) \
                    or (op == 'min' and val < best_val):
                best_id, best_item, best_val = id, i, val
        i += 1

    return best_item, best_id
