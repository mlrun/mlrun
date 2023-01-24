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
import random
import sys
from copy import deepcopy

import pandas as pd

from ..model import HyperParamOptions, RunObject, RunSpec
from ..utils import get_in

hyper_types = ["list", "grid", "random"]
default_max_iterations = 10
default_max_errors = 3


def get_generator(spec: RunSpec, execution, param_file_secrets: dict = None):
    options = spec.hyper_param_options
    strategy = spec.strategy or options.strategy
    if not spec.is_hyper_job() or strategy == "custom":
        return None
    options.validate()
    hyperparams = spec.hyperparams
    param_file = spec.param_file or options.param_file
    if strategy and strategy not in hyper_types:
        raise ValueError(f"unsupported hyper params strategy  ({strategy})")

    if param_file and hyperparams:
        raise ValueError("hyperparams and param_file cannot be used together")

    options.selector = options.selector or spec.selector
    if options.selector:
        parse_selector(options.selector)

    obj = None
    if param_file:
        obj = execution.get_dataitem(param_file, secrets=param_file_secrets)
        if not strategy and obj.suffix == ".csv":
            strategy = "list"
        if not strategy or strategy in ["grid", "random"]:
            hyperparams = json.loads(obj.get())

    if not strategy or strategy == "grid":
        return GridGenerator(hyperparams, options)

    if strategy == "random":
        return RandomGenerator(hyperparams, options)

    if obj:
        df = obj.as_df()
    else:
        df = pd.DataFrame(hyperparams)
    return ListGenerator(df, options)


class TaskGenerator:
    def __init__(self, options: HyperParamOptions):
        self.options = options

    def use_parallel(self):
        return self.options.parallel_runs or self.options.dask_cluster_uri

    @property
    def max_errors(self):
        return self.options.max_errors or default_max_errors

    @property
    def max_iterations(self):
        return self.options.max_iterations or default_max_iterations

    def generate(self, run: RunObject):
        pass

    def eval_stop_condition(self, results) -> bool:
        if not self.options.stop_condition:
            return False
        return eval(self.options.stop_condition, {}, results)


class GridGenerator(TaskGenerator):
    def __init__(self, hyperparams, options=None):
        super().__init__(options)
        self.hyperparams = hyperparams

    def generate(self, run: RunObject):
        i = 0
        params = self.grid_to_list()
        max = len(next(iter(params.values())))

        while i < max:
            newrun = get_run_copy(run)
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


class RandomGenerator(TaskGenerator):
    def __init__(self, hyperparams: dict, options=None):
        super().__init__(options)
        self.hyperparams = hyperparams

    def generate(self, run: RunObject):
        i = 0

        while i < self.max_iterations:
            newrun = get_run_copy(run)
            param_dict = newrun.spec.parameters or {}
            params = {k: random.sample(v, 1)[0] for k, v in self.hyperparams.items()}
            for key, values in params.items():
                param_dict[key] = values
            newrun.spec.parameters = param_dict
            newrun.metadata.iteration = i + 1
            i += 1
            yield newrun


class ListGenerator(TaskGenerator):
    def __init__(self, df, options=None):
        super().__init__(options)
        self.df = df

    def generate(self, run: RunObject):
        i = 0
        for _, row in self.df.iterrows():
            newrun = get_run_copy(run)
            param_dict = newrun.spec.parameters or {}
            for key, values in row.to_dict().items():
                param_dict[key] = values
            newrun.spec.parameters = param_dict
            newrun.metadata.iteration = i + 1
            i += 1
            yield newrun


def get_run_copy(run):
    newrun = deepcopy(run)
    newrun.spec.hyperparams = None
    newrun.spec.param_file = None
    newrun.spec.hyper_param_options = None
    return newrun


def parse_selector(criteria):
    idx = criteria.find(".")
    field = criteria
    if idx < 0:
        op = "max"
    else:
        op = criteria[:idx]
        field = criteria[idx + 1 :]
    if op not in ["min", "max"]:
        raise ValueError(
            f"illegal iteration selector {criteria}, "
            "selector format [min|max.]<result-name> e.g. max.accuracy"
        )
    return op, field


def selector(results: list, criteria):
    if not criteria:
        return 0, 0

    op, criteria = parse_selector(criteria)
    best_id = 0
    best_item = 0
    if op == "max":
        best_val = sys.float_info.min
    elif op == "min":
        best_val = sys.float_info.max

    i = 0
    for task in results:
        state = get_in(task, ["status", "state"])
        id = get_in(task, ["metadata", "iteration"])
        val = get_in(task, ["status", "results", criteria])
        if isinstance(val, str):
            try:
                val = float(val)
            except Exception:
                val = None
        if state != "error" and val is not None:
            if (op == "max" and val > best_val) or (op == "min" and val < best_val):
                best_id, best_item, best_val = id, i, val
        i += 1

    return best_item, best_id
