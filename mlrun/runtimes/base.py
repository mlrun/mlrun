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

import uuid
from ast import literal_eval
from datetime import datetime
import getpass
from copy import deepcopy
from os import environ

from ..datastore import StoreManager
from ..kfpops import write_kfpmeta, mlrun_op
from ..db import get_run_db, default_dbpath
from ..model import (
    RunObject, ModelObj, RunTemplate, BaseMetadata, ImageBuilder)
from ..secrets import SecretsStore
from ..utils import get_in, update_in, logger, is_ipython
from .utils import calc_hash, RunError, results_to_iter, default_image_name
from ..execution import MLClientCtx
from ..lists import RunList
from .generators import get_generator
from ..k8s_utils import get_k8s_helper
from ..config import config


class FunctionStatus(ModelObj):
    def __init__(self, state=None, build_pod=None):
        self.state = state
        self.build_pod = build_pod


class EntrypointParam(ModelObj):
    def __init__(self, name='', type=None, default=None, doc=''):
        self.name = name
        self.type = type
        self.default = default
        self.doc = doc


class FunctionEntrypoint(ModelObj):
    def __init__(
            self, name='', doc='', parameters=None, outputs=None, lineno=-1):
        self.name = name
        self.doc = doc
        self.parameters = [] if parameters is None else parameters
        self.outputs = [] if outputs is None else outputs
        self.lineno = lineno


class FunctionSpec(ModelObj):
    def __init__(self, command=None, args=None, image=None, mode=None,
                 build=None, entry_points=None, description=None):

        self.command = command or ''
        self.image = image or ''
        self.mode = mode
        self.args = args or []
        self.rundb = None
        self.description = description or ''

        self._build = None
        self.build = build
        # TODO: type verification (FunctionEntrypoint dict)
        self.entry_points = entry_points or {}

    @property
    def build(self) -> ImageBuilder:
        return self._build

    @build.setter
    def build(self, build):
        self._build = self._verify_dict(build, 'build', ImageBuilder)


class BaseRuntime(ModelObj):
    kind = 'base'
    _is_nested = False
    _is_remote = False
    _dict_fields = ['kind', 'metadata', 'spec', 'status']

    def __init__(self, metadata=None, spec=None):
        self._metadata = None
        self.metadata = metadata
        self.kfp = None
        self._spec = None
        self.spec = spec
        self._db_conn = None
        self._secrets = None
        self._k8s = None
        self._is_built = False
        self.is_child = False
        self._status = None
        self.status = None
        self._is_api_server = False

    def set_db_connection(self, conn, is_api=False):
        if not self._db_conn:
            self._db_conn = conn
        self._is_api_server = is_api

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

    @property
    def status(self) -> FunctionStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, 'status', FunctionStatus)

    def _get_k8s(self):
        return get_k8s_helper()

    def set_label(self, key, value):
        self.metadata.labels[key] = str(value)
        return self

    @property
    def is_deployed(self):
        return True

    def _is_remote_api(self):
        db = self._get_db()
        if db and db.kind == 'http':
            return True
        return False

    def _use_remote_api(self):
        if self._is_remote and not self.kfp and not self._is_api_server \
                and self._get_db() and self._get_db().kind == 'http':
            return True
        return False

    def _function_uri(self, tag=None):
        return '{}/{}:{}'.format(self.metadata.project, self.metadata.name,
                                 tag or self.metadata.tag or 'latest')

    def _get_db(self):
        if not self._db_conn:
            self.spec.rundb = self.spec.rundb or default_dbpath()
            if self.spec.rundb:
                self._db_conn = get_run_db(self.spec.rundb).connect(
                        self._secrets)
        return self._db_conn

    def run(self, runspec: RunObject = None, handler=None, name: str = '',
            project: str = '', params: dict = None, inputs: dict = None,
            out_path: str = '', workdir: str = '',
            watch: bool = True, schedule: str = ''):
        """Run a local or remote task.

        :param runspec:    run template object or dict (see RunTemplate)
        :param handler:    pointer or name of a function handler
        :param name:       execution name
        :param project:    project name
        :param params:     input parameters (dict)
        :param inputs:     input objects (dict of key: path)
        :param out_path:   default artifact output path
        :param workdir:    default input artifacts path
        :param watch:      watch/follow run log
        :param schedule:   cron string for scheduled jobs

        :return: run context object (dict) with run metadata, results and
            status
        """

        if runspec:
            runspec = deepcopy(runspec)
            if isinstance(runspec, str):
                runspec = literal_eval(runspec)

        if isinstance(runspec, RunTemplate):
            runspec = RunObject.from_template(runspec)
        if isinstance(runspec, dict) or runspec is None:
            runspec = RunObject.from_dict(runspec)

        runspec.spec.handler = handler or runspec.spec.handler or ''
        if runspec.spec.handler and self.kind not in ['handler', 'dask']:
            runspec.spec.handler = runspec.spec.handler_name

        runspec.metadata.name = name or runspec.metadata.name or \
            runspec.spec.handler_name or self.metadata.name
        runspec.metadata.project = project or runspec.metadata.project
        runspec.spec.parameters = params or runspec.spec.parameters
        runspec.spec.inputs = inputs or runspec.spec.inputs
        runspec.spec.output_path = out_path or runspec.spec.output_path
        runspec.spec.input_path = workdir or runspec.spec.input_path

        spec = runspec.spec
        if self.spec.mode and self.spec.mode == 'noctx':
            params = spec.parameters or {}
            for k, v in params.items():
                self.spec.args += ['--{}'.format(k), str(v)]

        if spec.secret_sources:
            self._secrets = SecretsStore.from_dict(spec.to_dict())

        # update run metadata (uid, labels) and store in DB
        meta = runspec.metadata
        meta.uid = meta.uid or uuid.uuid4().hex
        db = self._get_db()

        if not self.is_deployed:
            raise RunError(
                "function image is not built/ready, use .build() method first")

        if not self.is_child and self.kind != 'handler':
            dbstr = 'self' if self._is_api_server else self.spec.rundb
            logger.info('starting run {} uid={}  -> {}'.format(
                meta.name, meta.uid, dbstr))
            meta.labels['kind'] = self.kind
            meta.labels['owner'] = environ.get(
                    'V3IO_USERNAME', getpass.getuser())
            hashkey = calc_hash(self)
            if db:
                struct = self.to_dict()
                update_in(struct, 'metadata.tag', '')
                db.store_function(struct, self.metadata.name,
                                  self.metadata.project, hashkey)
                runspec.spec.function = self._function_uri(hashkey)

        # execute the job remotely (to a k8s cluster via the API service)
        if self._use_remote_api():
            if self._secrets:
                runspec.spec.secret_sources = self._secrets.to_serial()
            try:
                resp = db.submit_job(runspec, schedule=schedule)
                if schedule:
                    logger.info('task scheduled, {}'.format(resp))
                    return

                if resp:
                    txt = get_in(resp, 'status.status_text')
                    if txt:
                        logger.info(txt)
                if watch:
                    runspec.logs(True, self._get_db())
                    resp = self._get_db_run(runspec)
            except Exception as err:
                logger.error('got remote run err, {}'.format(err))
                result = self._post_run(task=runspec, err=err)
                return self._wrap_result(result, runspec, err=err)
            return self._wrap_result(resp, runspec)

        elif self._is_remote and not self._is_api_server and not self.kfp:
            logger.warning(
                'warning!, Api url not set, '
                'trying to exec remote runtime locally')

        execution = MLClientCtx.from_dict(runspec.to_dict(),
                                          db, autocommit=False)

        # create task generator (for child runs) from spec
        task_generator = None
        if not self._is_nested:
            task_generator = get_generator(spec, execution)

        last_err = None
        if task_generator:
            # multiple runs (based on hyper params or params file)
            generator = task_generator.generate(runspec)
            results = self._run_many(generator, execution, runspec)
            results_to_iter(results, runspec, execution)
            result = execution.to_dict()

        else:
            # single run
            try:
                resp = self._run(runspec, execution)
                if watch and self.kind not in ['', 'handler', 'local']:
                    state = runspec.logs(True, self._get_db())
                    if state != 'succeeded':
                        logger.warning('run ended with state {}'.format(state))
                result = self._post_run(resp, task=runspec)
            except RunError as err:
                last_err = err
                result = self._post_run(task=runspec, err=err)

        return self._wrap_result(result, runspec, err=last_err)

    def _wrap_result(self, result: dict, runspec: RunObject, err=None):

        if result and self.kfp and err is None:
            write_kfpmeta(result)

        # show ipython/jupyter result table widget
        results_tbl = RunList()
        if result:
            results_tbl.append(result)
        else:
            logger.info(
                'no returned result (job may still be in progress)')
            results_tbl.append(runspec.to_dict())
        if is_ipython and config.ipython_widget:
            results_tbl.show()

            uid = runspec.metadata.uid
            proj = '--project {}'.format(
                runspec.metadata.project) if runspec.metadata.project else ''
            print(
                'to track results use .show() or .logs() or in CLI: \n'
                '!mlrun get run {} {} , !mlrun logs {} {}'
                .format(uid, proj, uid, proj))

        if result:
            run = RunObject.from_dict(result)
            logger.info('run executed, status={}'.format(run.status.state))
            if run.status.state == 'error':
                if self._is_remote and not self.is_child:
                    print('runtime error: {}'.format(run.status.error))
                raise RunError(run.status.error)
            return run

        return None

    def _get_db_run(self, task: RunObject = None):
        if self._get_db() and task:
            project = task.metadata.project
            uid = task.metadata.uid
            iter = task.metadata.iteration
            return self._get_db().read_run(uid, project, iter=iter)
        if task:
            return task.to_dict()

    def _get_cmd_args(self, runobj, with_mlrun):
        extra_env = {'MLRUN_EXEC_CONFIG': runobj.to_json()}
        if self.spec.rundb:
            extra_env['MLRUN_DBPATH'] = self.spec.rundb or config.dbpath
        args = []
        command = self.spec.command
        code = self.spec.build.functionSourceCode \
            if hasattr(self.spec, 'build') else None

        if (code or runobj.spec.handler) and self.spec.mode == 'pass':
            raise ValueError('cannot use "pass" mode with code or handler')

        if code:
            extra_env['MLRUN_EXEC_CODE'] = code

        if with_mlrun:
            args = ['run', '--name', self.metadata.name, '--from-env']
            if not code:
                args += [command]
            command = 'mlrun'

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
                # self.store_run(task)
                resp = self._run(task, execution)
                resp = self._post_run(resp, task=task)
            except RunError as err:
                task.status.state = 'error'
                task.status.error = err
                resp = self._post_run(task=task, err=err)
            results.append(resp)
        return results

    def store_run(self, runobj: RunObject):
        if self._get_db() and runobj:
            project = runobj.metadata.project
            uid = runobj.metadata.uid
            iter = runobj.metadata.iteration
            self._get_db().store_run(runobj.to_dict(), uid, project, iter=iter)

    def _store_run_dict(self, rundict: dict):
        if self._get_db() and rundict:
            project = get_in(rundict, 'metadata.project', '')
            uid = get_in(rundict, 'metadata.uid')
            iter = get_in(rundict, 'metadata.iteration', 0)
            self._get_db().store_run(rundict, uid, project, iter=iter)

    def _post_run(
            self, resp: dict = None, task: RunObject = None, err=None) -> dict:
        """update the task state in the DB"""
        was_none = False
        if resp is None and task:
            was_none = True
            resp = self._get_db_run(task)

            if task.status.status_text:
                update_in(resp, 'status.status_text', task.status.status_text)

        if resp is None:
            return None

        if not isinstance(resp, dict):
            raise ValueError('post_run called with type {}'.format(type(resp)))

        updates = None
        last_state = get_in(resp, 'status.state', '')
        if last_state == 'error' or err:
            updates = {'status.last_update': str(datetime.now())}
            updates['status.state'] = 'error'
            update_in(resp, 'status.state', 'error')
            if err:
                update_in(resp, 'status.error', str(err))
            err = get_in(resp, 'status.error')
            if err:
                updates['status.error'] = str(err)
        elif not was_none and last_state != 'completed':
            updates = {'status.last_update': str(datetime.now())}
            updates['status.state'] = 'completed'
            update_in(resp, 'status.state', 'completed')

        if self._get_db() and updates:
            project = get_in(resp, 'metadata.project')
            uid = get_in(resp, 'metadata.uid')
            iter = get_in(resp, 'metadata.iteration', 0)
            self._get_db().update_run(updates, uid, project, iter=iter)

        return resp

    def _force_handler(self, handler):
        if not handler:
            raise RunError(
                'handler must be provided for {} runtime'.format(self.kind))

    def full_image_path(self, image=None):
        image = image or self.spec.image
        if not image.startswith('.'):
            return image
        if 'DEFAULT_DOCKER_REGISTRY' in environ:
            return '{}/{}'.format(
                environ.get('DEFAULT_DOCKER_REGISTRY'), image[1:])
        if 'IGZ_NAMESPACE_DOMAIN' in environ:
            return 'docker-registry.{}:80/{}'.format(
                environ.get('IGZ_NAMESPACE_DOMAIN'), image[1:])
        raise RunError('local container registry is not defined')

    def to_step(self, **kw):
        raise ValueError('.to_step() is deprecated, us .as_step() instead')

    def as_step(self, runspec: RunObject = None, handler=None, name: str = '',
                project: str = '', params: dict = None, hyperparams=None,
                selector='', inputs: dict = None, outputs: dict = None,
                in_path: str = '', out_path: str = '', image: str = ''):
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
        :param image:      container image to use

        :return: KubeFlow containerOp
        """

        if self.spec.image and not image:
            image = self.full_image_path()

        return mlrun_op(name, project, self,
                        runobj=runspec, handler=handler, params=params,
                        hyperparams=hyperparams, selector=selector,
                        inputs=inputs, outputs=outputs, job_image=image,
                        out_path=out_path, in_path=in_path)

    def export(self, target='', format='.yaml', secrets=None):
        """save function spec to a local/remote path (default to
        ./function.yaml)"""
        if self.kind == 'handler':
            raise ValueError('cannot export local handler function, use ' +
                             'code_to_function() to serialize your function')
        calc_hash(self)
        if format == '.yaml':
            data = self.to_yaml()
        else:
            data = self.to_json()
        stores = StoreManager(secrets)
        target = target or 'function.yaml'
        datastore, subpath = stores.get_or_create_store(target)
        datastore.put(subpath, data)
        logger.info('function spec saved to path: {}'.format(target))

    def save(self, tag='', versioned=True):
        db = self._get_db()
        if not db:
            logger.error('database connection is not configured')
            return

        tag = tag or self.metadata.tag or 'latest'
        self.metadata.tag = tag
        obj = self.to_dict()
        hashkey = calc_hash(self)
        logger.info('saving function: {}, tag: {}'.format(
            self.metadata.name, tag
        ))
        if versioned:
            db.store_function(obj, self.metadata.name,
                              self.metadata.project, hashkey)
        db.store_function(obj, self.metadata.name,
                          self.metadata.project, tag)
