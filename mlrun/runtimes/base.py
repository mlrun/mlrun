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
import inspect
import sys
import uuid
from ast import literal_eval
from datetime import datetime
import getpass
from copy import deepcopy
from os import environ

import pandas as pd
from io import StringIO

from ..datastore import StoreManager
from ..kfpops import write_kfpmeta, mlrun_op
from ..db import get_run_db, default_dbpath
from ..model import RunObject, ModelObj, RunTemplate, BaseMetadata, ImageBuilder
from ..secrets import SecretsStore
from ..utils import get_in, update_in, logger, is_ipython
from ..execution import MLClientCtx
from ..artifacts import TableArtifact
from ..lists import RunList
from .generators import get_generator
from ..k8s_utils import k8s_helper


class RunError(Exception):
    pass


class EntrypointParam(ModelObj):
    def __init__(self, type=None, default=None, help=None):
        self.type = type
        self.default = default
        self.help = help


class FunctionEntrypoint(ModelObj):
    def __init__(self, doc=None, parameters=None, outputs=None):
        self.doc = doc
        self.parameters = parameters or {}  # todo: type verification, EntrypointParam dict
        self.outputs = outputs or {}


class FunctionSpec(ModelObj):
    def __init__(self, command=None, args=None, image=None, mode=None,
                 build=None, entry_points=None, description=None):

        self.command = command or ''
        self.image = image or ''
        self.mode = mode or ''
        self.args = args or []
        self.rundb = default_dbpath()
        self.description = description or ''

        self._build = None
        self.build = build
        self.entry_points = entry_points or {}  # TODO: type verification (FunctionEntrypoint dict)

    @property
    def build(self) -> ImageBuilder:
        return self._build

    @build.setter
    def build(self, build):
        self._build = self._verify_dict(build, 'build', ImageBuilder)


class BaseRuntime(ModelObj):
    kind = 'base'
    _is_nested = False
    _dict_fields = ['kind', 'metadata', 'spec']

    def __init__(self, metadata=None, spec=None, build=None):
        self._metadata = None
        self.metadata = metadata
        self.kfp = None
        self._spec = None
        self.spec = spec
        self._db_conn = None
        self._secrets = None
        self._k8s = None
        self._is_built = False
        self.interactive = False

    @property
    def metadata(self) -> BaseMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, 'metadata', BaseMetadata)

    @property
    def spec(self) -> FunctionSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, 'spec', FunctionSpec)

    def _get_k8s(self):
        if not self._k8s:
            self._k8s = k8s_helper()
        return self._k8s

    def set_label(self, key, value):
        self.metadata.labels[key] = str(value)
        return self

    def run(self, runspec: RunObject = None, handler=None, name: str = '',
            project: str = '', params: dict = None, inputs: dict = None,
            out_path: str = '', visible: bool = True):
        """Run a local or remote task.

        :param runspec:    run template object or dict (see RunTemplate)
        :param handler:    pointer or name of a function handler
        :param name:       execution name
        :param project:    project name
        :param params:     input parameters (dict)
        :param inputs:     input objects (dict of key: path)
        :param out_path:   default artifact output path
        :param visible:    show run results in Jupyter

        :return: run context object (dict) with run metadata, results and status
        """

        def show(resp):
            results = RunList()
            # show ipython/jupyter result table widget
            if resp:
                results.append(resp)
            else:
                logger.info('no returned result (job may still be in progress)')
                results.append(runspec.to_dict())
            if is_ipython and visible:
                results.show()
            print('type result.show() to see detailed results/progress or use CLI:')
            uid = runspec.metadata.uid
            project = '--project {}'.format(runspec.metadata.project) if runspec.metadata.project else ''
            print('!mlrun get run --uid {} {}'.format(uid, project))
            return resp

        if runspec:
            runspec = deepcopy(runspec)
            if isinstance(runspec, str):
                runspec = literal_eval(runspec)

        if isinstance(runspec, RunTemplate):
            runspec = RunObject.from_template(runspec)
        if isinstance(runspec, dict) or runspec is None:
            runspec = RunObject.from_dict(runspec)
        runspec.metadata.name = name or runspec.metadata.name or self.metadata.name
        runspec.metadata.project = project or runspec.metadata.project
        runspec.spec.parameters = params or runspec.spec.parameters
        runspec.spec.inputs = inputs or runspec.spec.inputs
        runspec.spec.output_path = out_path or runspec.spec.output_path

        if handler and self.kind not in ['handler', 'dask']:
            if inspect.isfunction(handler):
                handler = handler.__name__
            else:
                handler = str(handler)
        runspec.spec.handler = handler or runspec.spec.handler

        spec = runspec.spec
        if self.spec.mode == 'noctx':
            params = spec.parameters or {}
            for k, v in params.items():
                self.spec.args += ['--{}'.format(k), str(v)]

        if spec.secret_sources:
            self._secrets = SecretsStore.from_dict(spec.to_dict())

        # update run metadata (uid, labels) and store in DB
        meta = runspec.metadata
        meta.uid = meta.uid or uuid.uuid4().hex
        logger.info('starting run {} uid={}  -> {}'.format(
            meta.name, meta.uid, self.spec.rundb))

        if self.spec.rundb:
            self._db_conn = get_run_db(self.spec.rundb).connect(self._secrets)

        meta.labels['kind'] = self.kind
        meta.labels['owner'] = meta.labels.get('owner', getpass.getuser())
        add_code_metadata(meta.labels)

        execution = MLClientCtx.from_dict(runspec.to_dict(),
                                          self._db_conn,
                                          autocommit=True)

        # form child run task generator from spec
        task_generator = None
        if not self._is_nested:
            task_generator = get_generator(spec, execution)

        if task_generator:
            # multiple runs (based on hyper params or params file)
            generator = task_generator.generate(runspec)
            results = self._run_many(generator, execution, runspec)
            self._results_to_iter(results, runspec, execution)
            resp = execution.to_dict()
            if resp and self.kfp:
                write_kfpmeta(resp)
            result = show(resp)
        else:
            # single run
            try:
                self.store_run(runspec)
                resp = self._run(runspec, execution)
                result = show(self._post_run(resp, task=runspec))
                if result and self.kfp:
                    write_kfpmeta(result)
            except RunError as err:
                logger.error(f'run error - {err}')
                result = show(self._post_run(task=runspec, err=err))

        if result:
            run = RunObject.from_dict(result)
            logger.info('run executed, status={}'.format(run.status.state))
            if run.status.state == 'error':
                raise RunError(run.status.error)
            return run

        return None

    def _get_db_run(self, task: RunObject = None):
        if self._db_conn and task:
            project = task.metadata.project
            uid = task.metadata.uid
            iter = task.metadata.iteration
            if iter:
                uid = '{}-{}'.format(uid, iter)
            return self._db_conn.read_run(uid, project)
        if task:
            return task.to_dict()

    def _get_cmd_args(self, runobj, with_mlrun):
        extra_env = {'MLRUN_EXEC_CONFIG': runobj.to_json()}
        if self.spec.rundb:
            extra_env['MLRUN_DBPATH'] = self.spec.rundb
        args = []
        command = self.spec.command
        if hasattr(self.spec, 'build'):
            code = self.spec.build.functionSourceCode
            if code:
                extra_env['MLRUN_EXEC_CODE'] = code
                if with_mlrun:
                    command = 'mlrun'
                    args = ['run', '--from-env']
        elif with_mlrun:
            command = 'mlrun'
            args = ['run', '--from-env', command]
        if runobj.spec.handler:
            args += ['--handler', runobj.spec.handler]
        if self.spec.args:
            args += self.spec.args
        return command, args, extra_env

    def _run(self, runspec: RunObject, execution) -> dict:
        pass

    def _run_many(self, tasks, execution, runobj: RunObject) -> RunList:
        results = RunList()
        for task in tasks:
            try:
                self.store_run(task)
                resp = self._run(task, execution)
                resp = self._post_run(resp, task=task)
            except RunError as err:
                task.status.state = 'error'
                task.status.error = err
                resp = self._post_run(task=task, err=err)
            results.append(resp)
        return results

    def store_run(self, runobj: RunObject, commit=True):
        if self._db_conn and runobj:
            project = runobj.metadata.project
            uid = runobj.metadata.uid
            iter = runobj.metadata.iteration
            if iter:
                uid = '{}-{}'.format(uid, iter)
            self._db_conn.store_run(runobj.to_dict(), uid, project, commit)

    def _post_run(self, resp: dict = None, task: RunObject = None, err=None):
        """update the task state in the DB"""
        was_none = False
        if resp is None and task:
            was_none = True
            resp = self._get_db_run(task)

        if resp is None:
            return None

        if not isinstance(resp, dict):
            raise ValueError('post_run called with type {}'.format(type(resp)))

        updates = None
        if get_in(resp, 'status.state', '') == 'error' or err:
            updates = {'status.last_update': str(datetime.now())}
            updates['status.state'] = 'error'
            update_in(resp, 'status.state', 'error')
            if err:
                update_in(resp, 'status.error', err)
            err = get_in(resp, 'status.error')
            if err:
                updates['status.error'] = err
        elif not was_none:
            updates = {'status.last_update': str(datetime.now())}
            updates['status.state'] = 'completed'
            update_in(resp, 'status.state', 'completed')

        if self._db_conn and updates:
            project = get_in(resp, 'metadata.project')
            uid = get_in(resp, 'metadata.uid')
            iter = get_in(resp, 'metadata.iteration', 0)
            if iter:
                uid = '{}-{}'.format(uid, iter)
            self._db_conn.update_run(updates, uid, project)

        return resp

    def _results_to_iter(self, results, runspec, execution):
        if not results:
            logger.error('got an empty results list in to_iter')
            return

        iter = []
        failed = 0
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

            self._post_run(task)
            iter.append(struct)

        df = pd.io.json.json_normalize(iter).sort_values('iter')
        header = df.columns.values.tolist()
        summary = [header] + df.values.tolist()
        item, id = selector(results, runspec.spec.selector)
        task = results[item] if id and results else None
        execution.log_iteration_results(id, summary, task)

        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False, line_terminator='\n', encoding='utf-8')
        execution.log_artifact(
            TableArtifact('iteration_results',
                          src_path='iteration_results.csv',
                          body=csv_buffer.getvalue(),
                          header=header,
                          viewer='table'))
        if failed:
            execution.set_state(error='{} tasks failed, check logs for db for details'.format(failed))
        else:
            execution.set_state('completed')

    def _force_handler(self, handler):
        if not handler:
            raise RunError('handler must be provided for {} runtime'.format(self.kind))

    def _image_path(self):
        image = self.spec.image
        if not image.startswith('.'):
            return image
        if 'DEFAULT_DOCKER_REGISTRY' in environ:
            return '{}/{}'.format(environ.get('DEFAULT_DOCKER_REGISTRY'), image[1:])
        if 'IGZ_NAMESPACE_DOMAIN' in environ:
            return 'docker-registry.{}:80/{}'.format(environ.get('IGZ_NAMESPACE_DOMAIN'), image[1:])
        raise RunError('local container registry is not defined')

    def to_step(self, runspec: RunObject = None, handler=None, name: str = '',
                project: str = '', params: dict = None, hyperparams=None, selector='',
                inputs: dict = None, outputs: dict = None,
                in_path: str = '', out_path: str = ''):
        """Run a local or remote task.

        :param runspec:    run template object or dict (see RunTemplate)
        :param handler:    name of the function handler
        :param name:       execution name
        :param project:    project name
        :param params:     input parameters (dict)
        :param hyperparams: hyper parameters
        :param selector:   selection criteria for hyper params
        :param inputs:     input objects (dict of key: path)
        :param outputs:    list of outputs which can pass in the workflow

        :return: KubeFlow containerOp
        """

        # expand local registry path, TODO: copy self to avoid modify the fn?
        self.spec.image = self._image_path()

        return mlrun_op(name, project, self,
                        runobj=runspec, handler=handler, params=params,
                        hyperparams=hyperparams, selector=selector,
                        inputs=inputs, outputs=outputs,
                        out_path=out_path, in_path=in_path)

    def export(self, target='', format='.yaml', secrets=None):
        """save function spec to a local/remote path (default to ./function.yaml)"""
        if self.kind == 'handler':
            raise ValueError('cannot export local handler function, use ' +
                             'code_to_function() to serialize your function')
        if format == '.yaml':
            data = self.to_yaml()
        else:
            data = self.to_json()
        stores = StoreManager(secrets)
        target = target or 'function.yaml'
        datastore, subpath = stores.get_or_create_store()
        datastore.put(subpath, data)
        logger.info('function spec saved to path: {}'.format(target))


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
            except:
                val = None
        if state != 'error' and val is not None:
            if (op == 'max' and val > best_val) \
                    or (op == 'min' and val < best_val):
                best_id, best_item, best_val = id, i, val
        i += 1

    return best_item, best_id


def add_code_metadata(labels):
    dirpath = './'
    try:
        from git import Repo
        from git.exc import GitCommandError, InvalidGitRepositoryError
    except ImportError:
        return

    try:
        repo = Repo(dirpath, search_parent_directories=True)
        remotes = [remote.url for remote in repo.remotes]
        if len(remotes) > 0:
            set_if_none(labels, 'repo', remotes[0])
            set_if_none(labels, 'commit', repo.head.commit.hexsha)
    except (GitCommandError, InvalidGitRepositoryError):
        pass


def set_if_none(struct, key, value):
    if not struct.get(key):
        struct[key] = value
