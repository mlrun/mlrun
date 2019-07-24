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
import socket
from datetime import datetime
import json
import uuid
import getpass
from copy import deepcopy
from os import environ, path
import pandas as pd

from ..db import get_run_db
from ..secrets import SecretsStore
from ..utils import run_keys, gen_md_table, dict_to_yaml, get_in, update_in


class RunError(Exception):
    pass


KFPMETA_DIR = environ.get('KFPMETA_OUT_DIR', '/')


def grid_to_list(params={}):
    arr = {}
    lastlen = 1
    for pk, pv in params.items():
        for p in arr.keys():
            arr[p] = arr[p] * len(pv)
        expanded = []
        for i in range(len(pv)):
            expanded += [pv[i]] * lastlen
        arr[pk] = expanded
        lastlen = lastlen * len(pv)

    return arr


def task_gen(struct, hyperparams):
    i = 0
    params = grid_to_list(hyperparams)
    max = len(next(iter(params.values())))

    while i < max:
        newstruct = deepcopy(struct)
        for key, values in params.items():
            newstruct['spec']['parameters'][key] = values[i]
        newstruct['metadata']['iteration'] = i + 1
        i += 1
        yield newstruct


class MLRuntime:
    kind = ''

    def __init__(self, command='', args=[], handler=None):
        self.struct = None
        self.command = command
        self.args = args
        self.handler = handler
        self.rundb = ''
        self.hyperparams = None
        self._secrets = None
        self.with_kfp = False

    def process_struct(self, struct):
        if 'status' not in struct:
            struct['status'] = {}
        if not get_in(struct, 'metadata.uid'):
            update_in(struct, 'metadata.uid', uuid.uuid4().hex)

        update_in(struct, 'metadata.labels.owner', getpass.getuser(), replace=False)
        update_in(struct, 'metadata.labels.host', socket.gethostname(), replace=False)
        update_in(struct, 'metadata.labels.runtime', self.kind)
        add_code_metadata(struct['metadata']['labels'])

        update_in(struct, 'spec.runtime.kind', self.kind)

        if self.command:
            update_in(struct, 'spec.runtime.command', self.command)
        else:
            self.command = get_in(struct, 'spec.runtime.command')
        if self.args:
            update_in(struct, 'spec.runtime.args', self.args)
        else:
            self.args = get_in(struct, 'spec.runtime.args', [])

        if 'parameters' not in struct['spec']:
            struct['spec']['parameters'] = {}
        self.struct = struct

    def _get_secrets(self):
        if not self._secrets:
            self._secrets = SecretsStore.from_dict(self.struct['spec'])
        return self._secrets

    def _save_run(self, struct):
        if self.rundb:
            rundb = get_run_db(self.rundb)
            rundb.connect(self._get_secrets())
            uid = struct['metadata']['uid']
            project = struct['metadata'].get('project', '')
            rundb.store_run(struct, uid, project, commit=True)
        return struct

    def run(self, hyperparams=None):
        self.hyperparams = hyperparams
        start = datetime.now()
        if self.hyperparams:
            results = self._run_many(task_gen(self.struct, hyperparams))
            self.struct['status'] = {'start_time': str(start)}
            self.struct['spec']['hyperparams'] = self.hyperparams
            results_to_iter_status(self.struct, results)
            resp = self.struct
        else:
            self.struct['status'] = {'start_time': str(start)}
            try:
                resp = self._run(self.struct)
            except RunError as err:
                self.struct['status']['state'] = 'error'
                self.struct['status']['error'] = err
                return self._post_run(self.struct)

        return self._post_run(resp)

    def _run(self, struct):
        pass

    def _run_many(self, tasks):
        results = []
        for task in tasks:
            try:
                resp = self._run(task)
                resp = self._post_run(resp)
            except RunError as err:
                task['status']['state'] = 'error'
                task['status']['error'] = err
                resp = self._post_run(task)
            results.append(resp)
        return results

    def _post_run(self, resp):
        if not resp:
            return {}

        if isinstance(resp, str):
            resp = json.loads(resp)

        if self.with_kfp:
            self.write_kfpmeta(resp)

        if resp['status'].get('state', '') != 'error':
            resp['status']['state'] = 'completed'
        resp['status']['last_update'] = str(datetime.now())
        self._save_run(resp)
        return resp

    def _force_handler(self):
        if not self.handler:
            raise RunError('handler must be provided for {} runtime'.format(self.kind))

    def write_kfpmeta(self, struct):
        outputs = struct['status'].get('outputs', {})
        metrics = {'metrics':
                       [{'name': k,
                         'numberValue': v,
                         } for k, v in outputs.items() if isinstance(v, (int, float, complex))]}
        with open(KFPMETA_DIR + 'mlpipeline-metrics.json', 'w') as f:
            json.dump(metrics, f)

        output_artifacts = get_kfp_outputs(
            struct['status'].get(run_keys.output_artifacts, []))

        text = '# Run Report\n'
        if 'iterations' in struct['status']:
            iter = struct['status']['iterations']
            with open(f'/tmp/iterations', 'w') as fp:
                fp.write(json.dumps(iter))
            iter_html = gen_md_table(iter[0], iter[1:])
            text += '## Iterations\n' + iter_html
            struct = deepcopy(struct)
            del struct['status']['iterations']

        text += "## Metadata\n```yaml\n" + dict_to_yaml(struct) + "```\n"

        #with open('sum.md', 'w') as fp:
        #    fp.write(text)

        metadata = {
            'outputs': output_artifacts + [{
                'type': 'markdown',
                'storage': 'inline',
                'source': text
            }]
        }
        with open(KFPMETA_DIR + 'mlpipeline-ui-metadata.json', 'w') as f:
            json.dump(metadata, f)


def get_kfp_outputs(artifacts):
    outputs = []
    for output in artifacts:
        key = output["key"]
        target = output.get('target_path', '')
        target = output.get('inline', target)
        try:
            with open(f'/tmp/{key}', 'w') as fp:
                fp.write(target)
        except:
            pass

        if target.startswith('v3io:///'):
            target = target.replace('v3io:///', 'http://v3io-webapi:8081/')

        viewer = output.get('viewer', '')
        if viewer in ['web-app', 'chart']:
            meta = {'type': 'web-app',
                    'source': target}
            outputs += [meta]

        elif viewer == 'table':
            header = output.get('header', None)
            if header and target.endswith('.csv'):
                meta = {'type': 'table',
                        'format': 'csv',
                        'header': header,
                        'source': target}
                outputs += [meta]

    return outputs


def results_to_iter_status(base_struct, results):
    iter = []
    for task in results:
        struct = {'param': task['spec'].get('parameters', {}),
                  'output': task['status'].get('outputs', {}),
                  'state': task['status'].get('state'),
                  'iter': task['metadata'].get('iteration'),
                  }
        iter.append(struct)

    df = pd.io.json.json_normalize(iter).sort_values('iter')
    iter_table = [df.columns.values.tolist()] + df.values.tolist()
    base_struct['status']['iterations'] = iter_table


# class MLRunChild:
#
#     def __init__(self, task):
#         self.param = task['spec'].get('parameters', {})
#         self.output = task['status'].get('outputs', {})
#         self.state = task['status'].get('state')
#         self.iter = task['metadata'].get('iteration')
#
#     def as_dict(self):
#         result = {'iter': self.iter, 'state': self.state}
#         for k, v in self.param.items():
#             result[f'param.{k}'] = v
#         for k, v in self.output.items():
#             result[f'output.{k}'] = v
#         return result
#
#
# class MLRunChild(list):
#
#     def to_table(self):
#         df = pd.DataFrame([i.as_dict() for i in self]).sort_values('iter')
#         return [df.columns.values.tolist()] + df.values.tolist()


def add_code_metadata(labels):
    dirpath = './'
    try:
        filepath = __file__
        if path.isfile(filepath):
            labels['file'] = filepath
            dirpath = path.dirname(filepath)
    except NameError:
        pass

    try:
        from git import Repo
        from git.exc import GitCommandError, InvalidGitRepositoryError
    except ImportError:
        return labels

    try:
        repo = Repo(dirpath, search_parent_directories=True)
        remotes = [remote.url for remote in repo.remotes]
        if len(remotes) > 0:
            labels['repo'] = remotes[0]
            labels['commit'] = repo.head.commit.hexsha
    except (GitCommandError, InvalidGitRepositoryError):
        pass
    return labels


