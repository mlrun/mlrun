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
import hashlib
from copy import deepcopy
from sys import stderr
import pandas as pd
from io import StringIO
from ..utils import logger
from ..config import config
from .generators import selector
from ..utils import get_in
from ..artifacts import TableArtifact
from kubernetes import client


class RunError(Exception):
    pass


mlrun_key = 'mlrun/'


def calc_hash(func, tag=''):
    # remove tag, hash, date from calculation
    tag = tag or func.metadata.tag
    status = func.status
    func.metadata.tag = ''
    func.metadata.hash = ''
    func.status = None
    func.metadata.updated = None

    data = func.to_json().encode()
    h = hashlib.sha1()
    h.update(data)
    hashkey = h.hexdigest()
    func.metadata.tag = tag
    func.metadata.hash = hashkey
    func.status = status
    return hashkey


def log_std(db, runobj, out, err='', skip=False, show=True):
    line = '> ' + '-' * 15 + ' Iteration: ({}) ' + '-' * 15 + '\n'
    if out:
        iter = runobj.metadata.iteration
        if iter:
            out = line.format(iter) + out
        if show:
            print(out)
        if db and not skip:
            uid = runobj.metadata.uid
            project = runobj.metadata.project or ''
            db.store_log(uid, project, out.encode(), append=True)
    if err:
        logger.error('exec error - {}'.format(err))
        print(err, file=stderr)
        raise RunError(err)


class AsyncLogWriter:
    def __init__(self, db, runobj):
        self.db = db
        self.uid = runobj.metadata.uid
        self.project = runobj.metadata.project or ''
        self.iter = runobj.metadata.iteration

    def write(self, data):
        if self.db:
            self.db.store_log(self.uid, self.project, data, append=True)

    def flush(self):
        # todo: verify writes are large enough, if not cache and use flush
        pass


def add_code_metadata():
    dirpath = './'
    try:
        from git import Repo
        from git.exc import GitCommandError, InvalidGitRepositoryError
    except ImportError:
        return None

    try:
        repo = Repo(dirpath, search_parent_directories=True)
        remotes = [remote.url for remote in repo.remotes]
        if len(remotes) > 0:
            return '{}#{}'.format(remotes[0], repo.head.commit.hexsha)
    except (GitCommandError, InvalidGitRepositoryError):
        pass
    return None


def set_if_none(struct, key, value):
    if not struct.get(key):
        struct[key] = value


def results_to_iter(results, runspec, execution):
    if not results:
        logger.error('got an empty results list in to_iter')
        return

    iter = []
    failed = 0
    running = 0
    for task in results:
        state = get_in(task, ['status', 'state'])
        id = get_in(task, ['metadata', 'iteration'])
        struct = {'param': get_in(task, ['spec', 'parameters'], {}),
                  'output': get_in(task, ['status', 'results'], {}),
                  'state': state,
                  'iter': id,
                  }
        if state == 'error':
            failed += 1
            err = get_in(task, ['status', 'error'], '')
            logger.error('error in task  {}:{} - {}'.format(
                runspec.metadata.uid, id, err))
        elif state != 'completed':
            running += 1

        iter.append(struct)

    df = pd.io.json.json_normalize(iter).sort_values('iter')
    header = df.columns.values.tolist()
    summary = [header] + df.values.tolist()
    item, id = selector(results, runspec.spec.selector)
    task = results[item] if id and results else None
    execution.log_iteration_results(id, summary, task)

    csv_buffer = StringIO()
    df.to_csv(
        csv_buffer, index=False, line_terminator='\n', encoding='utf-8')
    execution.log_artifact(
        TableArtifact('iteration_results',
                      src_path='iteration_results.csv',
                      body=csv_buffer.getvalue(),
                      header=header,
                      viewer='table'))
    if failed:
        execution.set_state(
            error='{} tasks failed, check logs in db for details'.format(
                failed), commit=False)
    elif running == 0:
        execution.set_state('completed', commit=False)
    execution.commit()


def default_image_name(function):
    meta = function.metadata
    proj = meta.project or config.default_project
    return '.mlrun/func-{}-{}-{}'.format(proj, meta.name, meta.tag or 'latest')


def set_named_item(obj, item):
    if isinstance(item, dict):
        obj[item['name']] = item
    else:
        obj[item.name] = item


def get_item_name(item, attr='name'):
    if isinstance(item, dict):
        return item[attr]
    else:
        return getattr(item, attr, None)


def apply_kfp(modify, cop, runtime):
    modify(cop)
    api = client.ApiClient()
    for k, v in cop.pod_labels.items():
        runtime.metadata.labels[k] = v
    for k, v in cop.pod_annotations.items():
        runtime.metadata.annotations[k] = v
    if cop.container.env:
        [runtime.spec.env.append(e)
         for e in api.sanitize_for_serialization(cop.container.env)]
        cop.container.env.clear()

    if cop.volumes and cop.container.volume_mounts:
        vols = api.sanitize_for_serialization(
            cop.volumes)
        mounts = api.sanitize_for_serialization(
            cop.container.volume_mounts)
        runtime.spec.update_vols_and_mounts(vols, mounts)
        cop.volumes.clear()
        cop.container.volume_mounts.clear()

    return runtime


def get_resource_labels(function, uid=None):
    meta = function.metadata
    labels = deepcopy(meta.labels)
    labels[mlrun_key + 'class'] = function.kind
    labels[mlrun_key + 'project'] = meta.project
    labels[mlrun_key + 'function'] = '{}'.format(meta.name)
    labels[mlrun_key + 'tag'] = '{}'.format(meta.tag or 'latest')

    if uid:
        labels[mlrun_key + 'uid'] = uid

    return labels


def get_func_selector(project, name=None, tag=None):
    s = ['{}project={}'.format(mlrun_key, project)]
    if name:
        s.append('{}function={}'.format(mlrun_key, name))
        s.append('{}tag={}'.format(mlrun_key, tag or 'latest'))
    return s


class k8s_resource:
    kind = ''
    per_run = False
    per_function = False
    k8client = None

    def deploy_function(self, function):
        pass

    def release_function(self, function):
        pass

    def submit_run(self, function, runobj):
        pass

    def get_object(self, name, namespace=None):
        return None

    def get_status(self, name, namespace=None):
        return None

    def del_object(self, name, namespace=None):
        pass

    def list_objects(self, namespace=None, selector=[], states=None):
        return []

    def get_pods(self, name, namespace=None, master=False):
        return {}

    def clean_objects(self, namespace=None, selector=[], states=None):
        if not selector and not states:
            raise ValueError(
                'labels selector or states list must be specified')
        items = self.list_objects(namespace, selector, states)
        for item in items:
            self.del_object(item.metadata.name, item.metadata.namespace)

