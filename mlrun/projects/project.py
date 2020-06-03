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

import shutil

from tabulate import tabulate

from ..db import get_run_db
from ..artifacts import ArtifactManager, ArtifactProducer, dict_to_artifact
from ..secrets import SecretsStore
from ..model import ModelObj
import tarfile
from tempfile import mktemp
from git import Repo

import yaml
from os import path, remove

from ..datastore import StoreManager
from ..config import config
from ..run import (import_function, code_to_function, new_function,
                   download_object, run_pipeline, get_object, wait_for_pipeline_completion)
import importlib.util as imputil
from urllib.parse import urlparse
from kfp import compiler

from ..utils import update_in, new_pipe_meta, logger, dict_to_str, is_ipython, RunNotifications
from ..runtimes.utils import add_code_metadata


class ProjectError(Exception):
    pass


def new_project(name, context=None, init_git=False):
    """Create a new MLRun project

    :param name:       project name
    :param context:    project local directory path
    :param init_git:   if True, will git init the context dir

    :returns: project object
    """
    project = MlrunProject(name=name)
    project.context = context

    if init_git:
        repo = Repo.init(context)
        project.repo = repo

    return project


def load_project(context, url=None, name=None, secrets=None,
                 init_git=False, subpath='', clone=False):
    """Load an MLRun project from git or tar or dir

    :param context:    project local directory path
    :param url:        git or tar.gz sources archive path e.g.:
                       git://github.com/mlrun/demo-xgb-project.git
    :param name:       project name
    :param secrets:    key:secret dict or SecretsStore used to download sources
    :param init_git:   if True, will git init the context dir
    :param subpath:    project subpath (within the archive)
    :param clone:      if True, always clone (delete any existing content)

    :returns: project object
    """

    secrets = secrets or {}
    repo = None
    project = None
    if url:
        if url.endswith('.yaml'):
            project = _load_project_file(url, name, secrets)
        elif url.startswith('git://'):
            url, repo = clone_git(url, context, secrets, clone)
        elif url.endswith('.tar.gz'):
            clone_tgz(url, context, secrets)
        else:
            raise ValueError('unsupported code archive {}'.format(url))

    else:
        if not path.isdir(context):
            raise ValueError('context {} is not an existing dir path'.format(
                context))
        try:
            repo = Repo(context)
            url = _get_repo_url(repo)
        except Exception:
            if init_git:
                repo = Repo.init(context)

    if not project:
        project = _load_project_dir(context, name, subpath)
    project.source = url
    project.repo = repo
    if repo:
        project.branch = repo.active_branch.name
    project.origin_url = url
    project.register_artifacts()
    return project


def _load_project_dir(context, name='', subpath=''):
    fpath = path.join(context, subpath, 'project.yaml')
    if path.isfile(fpath):
        with open(fpath) as fp:
            data = fp.read()
            struct = yaml.load(data, Loader=yaml.FullLoader)
            struct['context'] = context
            struct['name'] = name or struct.get('name', '')
            project = MlrunProject.from_dict(struct)

    elif path.isfile(path.join(context, subpath, 'function.yaml')):
        func = import_function(path.join(context, subpath, 'function.yaml'))
        project = MlrunProject(name=func.metadata.project,
                               functions=[{'url': 'function.yaml',
                                           'name': func.metadata.name}])
    else:
        raise ValueError('project or function YAML not found in path')

    project.context = context
    project.name = name or project.name
    project.subpath = subpath
    return project


def _load_project_file(url, name='', secrets=None):
    try:
        obj = get_object(url, secrets)
    except FileNotFoundError as e:
        raise FileNotFoundError('cant find project file at {}, {}'.format(url, e))
    struct = yaml.load(obj, Loader=yaml.FullLoader)
    struct['name'] = name or struct.get('name', '')
    project = MlrunProject.from_dict(struct)
    return project


class MlrunProject(ModelObj):
    kind = 'project'

    def __init__(self, name=None, description=None, params=None,
                 functions=None, workflows=None, artifacts=None,
                 artifact_path=None, conda=None):

        self._initialized = False
        self.name = name
        self.description = description
        self.tag = ''
        self.origin_url = ''
        self._source = ''
        self.context = None
        self.subpath = ''
        self.branch = None
        self.repo = None
        self._secrets = SecretsStore()
        self.params = params or {}
        self.conda = conda or {}
        self._mountdir = None
        self._artifact_mngr = None
        self.artifact_path = artifact_path

        self.workflows = workflows or []
        self.artifacts = artifacts or []

        self._function_objects = {}
        self._function_defs = {}
        self.functions = functions or []

    @property
    def source(self) -> str:
        """source url or git repo"""
        if not self._source:
            if self.repo:
                url = _get_repo_url(self.repo)
                if url:
                    self._source = url

        return self._source

    @source.setter
    def source(self, src):
        self._source = src

    def _source_repo(self):
        src = self.source
        if src:
            return src.split('#')[0]
        return ''

    def _get_hexsha(self):
        try:
            if self.repo:
                return self.repo.head.commit.hexsha
        except Exception:
            pass
        return None

    @property
    def mountdir(self) -> str:
        """specify to mount the context dir inside the function container
        use '.' to use the same path as in the client e.g. Jupyter"""

        if self._mountdir and self._mountdir in ['.', './']:
            return path.abspath(self.context)
        return self._mountdir

    @mountdir.setter
    def mountdir(self, mountdir):
        self._mountdir = mountdir

    @property
    def functions(self) -> list:
        """list of function object/specs used in this project"""
        funcs = []
        for name, f in self._function_defs.items():
            if hasattr(f, 'to_dict'):
                spec = f.to_dict(strip=True)
                if f.spec.build.source and \
                        f.spec.build.source.startswith(self._source_repo()):
                    update_in(spec, 'spec.build.source', './')
                funcs.append({'name': name,
                              'spec': spec})
            else:
                funcs.append(f)
        return funcs

    @functions.setter
    def functions(self, funcs):
        if not isinstance(funcs, list):
            raise ValueError('functions must be a list')

        func_defs = {}
        for f in funcs:
            if not isinstance(f, dict) and not hasattr(f, 'to_dict'):
                raise ValueError('functions must be an objects or dict')
            if isinstance(f, dict):
                name = f.get('name', '')
                if not name:
                    raise ValueError('function name must be specified in dict')
            else:
                name = f.metadata.name
            func_defs[name] = f

        self._function_defs = func_defs

    @property
    def workflows(self) -> list:
        """list of workflows specs used in this project"""
        return [w for w in self._workflows.values()]

    @workflows.setter
    def workflows(self, workflows):
        if not isinstance(workflows, list):
            raise ValueError('workflows must be a list')

        wfdict = {}
        for w in workflows:
            if not isinstance(w, dict):
                raise ValueError('workflow must be a dict')
            name = w.get('name', '')
            # todo: support steps dsl as code alternative
            if not name:
                raise ValueError('workflow "name" must be specified')
            if 'path' not in w and 'code' not in w:
                raise ValueError('workflow source "path" or "code" must be specified')
            wfdict[name] = w

        self._workflows = wfdict

    def set_workflow(self, name, workflow_path: str, embed=False, **args):
        """add or update a workflow, specify a name and the code path"""
        if not workflow_path:
            raise ValueError('valid workflow_path must be specified')
        if embed:
            if self.context and not workflow_path.startswith('/'):
                workflow_path = path.join(self.context, workflow_path)
            with open(workflow_path, 'r') as fp:
                txt = fp.read()
            workflow = {'name': name, 'code': txt}
        else:
            workflow = {'name': name, 'path': workflow_path}
        if args:
            workflow['args'] = args
        self._workflows[name] = workflow

    @property
    def artifacts(self) -> list:
        """list of artifacts used in this project"""
        return [a for a in self._artifacts.values()]

    @artifacts.setter
    def artifacts(self, artifacts):
        if not isinstance(artifacts, list):
            raise ValueError('artifacts must be a list')

        afdict = {}
        for a in artifacts:
            if not isinstance(a, dict) and not hasattr(a, 'to_dict'):
                raise ValueError('artifacts must be a dict or class')
            if isinstance(a, dict):
                key = a.get('key', '')
                if not key:
                    raise ValueError('artifacts "key" must be specified')
            else:
                key = a.key
                a = a.to_dict()

            afdict[key] = a

        self._artifacts = afdict

    def _get_artifact_mngr(self):
        if self._artifact_mngr:
            return self._artifact_mngr
        db = get_run_db().connect(self._secrets)
        sm = StoreManager(self._secrets, db)
        self._artifact_mngr = ArtifactManager(sm, db)
        return self._artifact_mngr

    def register_artifacts(self):
        """register the artifacts in the MLRun DB (under this project)"""
        am = self._get_artifact_mngr()
        producer = ArtifactProducer('project', self.name, self.name,
                                    tag=self._get_hexsha() or 'latest')
        for name, obj in self._artifacts.items():
            artifact = dict_to_artifact(obj)
            am.log_artifact(producer, artifact, upload=False)

    def log_artifact(self, item, body=None, tag='', local_path='',
                     artifact_path=None, format=None, upload=True,
                     labels=None, target_path=None):
        am = self._get_artifact_mngr()
        artifact_path = artifact_path or self.artifact_path
        producer = ArtifactProducer('project', self.name, self.name,
                                    tag=self._get_hexsha() or 'latest')
        item = am.log_artifact(producer, item, body, tag=tag,
                               local_path=local_path,
                               artifact_path=artifact_path, format=format,
                               upload=upload, labels=labels,
                               target_path=target_path)
        self._artifacts[item.key] = item.base_dict()

    def reload(self, sync=False):
        """reload the project and function objects from yaml/specs

        :param sync:  set to True to load functions objects

        :returns: project object
        """
        if self.context:
            project = _load_project_dir(self.context, self.name, self.subpath)
        else:
            project = _load_project_file(self.origin_url, self.name, self._secrets)
        project.source = self.source
        project.repo = self.repo
        project.branch = self.branch
        project.origin_url = self.origin_url
        if sync:
            project.sync_functions()
        self.__dict__.update(project.__dict__)
        return project

    def set_function(self, func, name='', kind='', image=None, with_repo=None):
        """update or add a function object to the project

        function can be provided as an object (func) or a .py/.ipynb/.yaml url
        support url prefixes:
            object (s3://, v3io://, ..)
            MLRun DB e.g. db://project/func:ver
            functions hub/market: e.g. hub://sklearn_classifier:master

        examples:

        proj.set_function(func_object)
        proj.set_function('./src/mycode.py', 'ingest',
                          image='myrepo/ing:latest', with_repo=True)
        proj.set_function('http://.../mynb.ipynb', 'train')
        proj.set_function('./func.yaml')
        proj.set_function('hub://get_toy_data', 'getdata')

        :param func:      function object or spec/code url
        :param name:      name of the function (under the project)
        :param kind:      runtime kind e.g. job, nuclio, spark, dask, mpijob
                          default: job
        :param image:     docker image to be used, can also be specified in
                          the function object/yaml
        :param with_repo: add (clone) the current repo to the build source

        :returns: project object
        """
        if isinstance(func, str):
            if not name:
                raise ValueError('function name must be specified')
            fdict = {'url': func, 'name': name, 'kind': kind,
                     'image': image, 'with_repo': with_repo}
            func = {k: v for k, v in fdict.items() if v}
            name, f = _init_function_from_dict(func, self)
        elif hasattr(func, 'to_dict'):
            name, f = _init_function_from_obj(func, self, name=name)
            if image:
                f.spec.image = image
            if with_repo:
                f.spec.build.source = './'

            if not name:
                raise ValueError('function name must be specified')
        else:
            raise ValueError('func must be a function url or object')

        self._function_defs[name] = func
        self._function_objects[name] = f
        return f

    def func(self, key, sync=False):
        """get function object by name

        :param sync:  will reload/reinit the function

        :returns: function object
        """
        if key not in self._function_defs:
            raise KeyError('function {} not found'.format(key))
        if sync or not self._initialized or key not in self._function_objects:
            self.sync_functions()
        return self._function_objects[key]

    def pull(self, branch=None, remote=None):
        """pull/update sources from git or tar into the context dir

        :param branch:  git branch, if not the current one
        :param remote:  git remote, if other than origin
        """
        url = self.origin_url
        if url and url.startswith('git://'):
            if not self.repo:
                raise ValueError('repo was not initialized, use load_project()')
            branch = branch or self.repo.active_branch.name
            remote = remote or 'origin'
            self.repo.git.pull(remote, branch)
        elif url and url.endswith('.tar.gz'):
            if not self.context:
                raise ValueError('target dit (context) is not set')
            clone_tgz(url, self.context, self._secrets)

    def create_remote(self, url, name='origin'):
        """create remote for the project git

        :param url:   remote git url
        :param name:  name for the remote (default is 'origin')
        """
        if not self.repo:
            raise ValueError('git repo is not set/defined')
        self.repo.create_remote(name, url=url)
        url = url.replace('https://', 'git://')
        try:
            url = '{}#refs/heads/{}'.format(url, self.repo.active_branch.name)
        except Exception:
            pass
        self._source = self.source or url
        self.origin_url = self.origin_url or url

    def push(self, branch, message=None, update=True, remote=None,
             add: list = None):
        """update spec and push updates to remote git repo

         :param branch:  target git branch
         :param message: git commit message
         :param update:  update files (git add update=True)
         :param remote:  git remote, default to origin
         :param add:     list of files to add
         """
        repo = self.repo
        if not repo:
            raise ValueError('git repo is not set/defined')
        self.save()

        add = add or []
        add.append('project.yaml')
        repo.index.add(add)
        if update:
            repo.git.add(update=True)
        if repo.is_dirty():
            if not message:
                raise ValueError('please specify the commit message')
            repo.git.commit(m=message)

        if not branch:
            raise ValueError('please specify the remote branch')
        repo.git.push(remote or 'origin', branch)

    def sync_functions(self, names: list = None, always=True, save=False):
        """reload function objects from specs and files"""
        if self._initialized and not always:
            return

        funcs = {}
        if not names:
            names = self._function_defs.keys()
        origin = add_code_metadata(self.context)
        for name in names:
            f = self._function_defs.get(name)
            if not f:
                raise ValueError('function named {} not found'.format(name))
            if hasattr(f, 'to_dict'):
                name, func = _init_function_from_obj(f, self)
            else:
                if not isinstance(f, dict):
                    raise ValueError('function must be an object or dict')
                name, func = _init_function_from_dict(f, self)
            func.spec.build.code_origin = origin
            funcs[name] = func
            if save:
                func.save(versioned=False)

        self._function_objects = funcs
        self._initialized = True

    def with_secrets(self, kind, source, prefix=''):
        """register a secrets source (file, env or dict)

        read secrets from a source provider to be used in workflows, e.g.

        proj.with_secrets('file', 'file.txt')
        proj.with_secrets('inline', {'key': 'val'})
        proj.with_secrets('env', 'ENV1,ENV2', prefix='PFX_')

        :param kind:   secret type (file, inline, env)
        :param source: secret data or link (see example)
        :param prefix: add a prefix to the keys in this source

        :returns: project object
        """
        self._secrets.add_source(kind, source, prefix)
        return self

    def get_secret(self, key: str):
        """get a key based secret e.g. DB password from the context
        secrets can be specified when invoking a run through files, env, ..
        """
        if self._secrets:
            return self._secrets.get(key)
        return None

    def get_param(self, key: str, default=None):
        """get project param by key"""
        if self.params:
            return self.params.get(key)
        return None

    def run(self, name=None, workflow_path=None, arguments=None,
            artifact_path=None, namespace=None, sync=False,
            watch=False, dirty=False, ttl=None):
        """run a workflow using kubeflow pipelines

        :param name:      name of the workflow
        :param workflow_path:
                          url to a workflow file, if not a project workflow
        :param arguments:
                          kubeflow pipelines arguments (parameters)
        :param artifact_path:
                          target path/url for workflow artifacts, the string
                          '{{workflow.uid}}' will be replaced by workflow id
        :param namespace: kubernetes namespace if other than default
        :param sync:      force functions sync before run
        :param watch:     wait for pipeline completion
        :param dirty:     allow running the workflow when the git repo is dirty
        :param ttl        pipeline ttl in secs (after that the pods will be removed)

        :returns: run id
        """

        need_repo = self._need_repo()
        if self.repo and self.repo.is_dirty():
            msg = 'you seem to have uncommitted git changes, use .push()'
            if dirty or not need_repo:
                logger.warning('WARNING!, ' + msg)
            else:
                raise ProjectError(msg + ' or dirty=True')

        if need_repo and self.repo and not self.source:
            raise ProjectError(
                'remote repo is not defined, use .create_remote() + push()')

        self.sync_functions(always=sync)
        if not self._function_objects:
            raise ValueError('no functions in the project')

        if not name and not workflow_path:
            if self._workflows:
                name = list(self._workflows.keys())[0]
            else:
                raise ValueError('workflow name or path must be specified')

        code = None
        if not workflow_path:
            if name not in self._workflows:
                raise ValueError('workflow {} not found'.format(name))
            workflow_path, code, arguments = self._get_wf_cfg(name, arguments)

        name = '{}-{}'.format(self.name, name) if name else self.name
        artifact_path = artifact_path or self.artifact_path
        run = _run_pipeline(self, name, workflow_path, self._function_objects,
                            secrets=self._secrets, arguments=arguments,
                            artifact_path=artifact_path, namespace=namespace,
                            ttl=ttl)
        if code:
            remove(workflow_path)
        if watch:
            self.get_run_status(run,
                                notifiers=RunNotifications(with_slack=True))
        return run

    def _get_wf_cfg(self, name, arguments=None):
        workflow = self._workflows.get(name)
        code = workflow.get('code')
        if code:
            workflow_path = mktemp('.py')
            with open(workflow_path, 'w') as wf:
                wf.write(code)
        else:
            workflow_path = workflow.get('path', '')
            if self.context and not workflow_path.startswith('/'):
                workflow_path = path.join(self.context, workflow_path)

        wf_args = workflow.get('args', {})
        if arguments:
            for k, v in arguments.items():
                wf_args[k] = v

        return workflow_path, code, wf_args

    def _need_repo(self):
        for f in self._function_objects.values():
            if f.spec.build.source in ['.', './']:
                return True
        return False

    def save_workflow(self, name, target, artifact_path=None, ttl=None):
        """create and save a workflow as a yaml or archive file

        :param name:   workflow name
        :param target: target file path (can end with .yaml or .zip)
        :param artifact_path:
                       target path/url for workflow artifacts, the string
                       '{{workflow.uid}}' will be replaced by workflow id
        :param ttl     pipeline ttl in secs (after that the pods will be removed)
        """
        if not name or name not in self._workflows:
            raise ValueError('workflow {} not found'.format(name))

        workflow_path, code, _ = self._get_wf_file(name)
        pipeline = _create_pipeline(self, workflow_path, self._function_objects,
                                    secrets=self._secrets)

        artifact_path = artifact_path or self.artifact_path
        conf = new_pipe_meta(artifact_path, ttl=ttl)
        compiler.Compiler().compile(pipeline, target, pipeline_conf=conf)
        if code:
            remove(workflow_path)

    def get_run_status(self, workflow_id, timeout=60 * 60,
                       expected_statuses=None,
                       notifiers: RunNotifications = None):
        status = ''
        if timeout:
            logger.info('waiting for pipeline run completion')
            run_info = wait_for_pipeline_completion(
                workflow_id, timeout=timeout,
                expected_statuses=expected_statuses)
            if run_info:
                status = run_info['run'].get('status')

        mldb = get_run_db().connect(self._secrets)
        runs = mldb.list_runs(project=self.name, labels=f'workflow={workflow_id}')

        had_errors = 0
        for r in runs:
            if r['status'].get('state', '') == 'error':
                had_errors += 1

        text = f'Workflow {workflow_id} finished'
        if had_errors:
            text += f' with {had_errors} errors'
        if status:
            text += f', status={status}'

        if notifiers:
            notifiers.push(text, runs)
        return status, had_errors, text

    def clear_context(self):
        """delete all files and clear the context dir"""
        if self.context and path.exists(self.context) and path.isdir(self.context):
            shutil.rmtree(self.context)

    def save(self, filepath=None):
        """save the project object into a file (default to project.yaml)"""
        filepath = filepath or path.join(self.context, self.subpath,
                                         'project.yaml')
        with open(filepath, 'w') as fp:
            fp.write(self.to_yaml())


def _init_function_from_dict(f, project):
    name = f.get('name', '')
    url = f.get('url', '')
    kind = f.get('kind', '')
    image = f.get('image', None)
    with_repo = f.get('with_repo', False)

    if with_repo and not project.source:
        raise ValueError('project source must be specified when cloning context')

    in_context = False
    if not url and 'spec' not in f:
        raise ValueError('function missing a url or a spec')

    if url and '://' not in url:
        if project.context and not url.startswith('/'):
            url = path.join(project.context, url)
            in_context = True
        if not path.isfile(url):
            raise OSError('{} not found'.format(url))

    if 'spec' in f:
        func = new_function(name, runtime=f['spec'])
    elif url.endswith('.yaml') or url.startswith('db://') \
            or url.startswith('hub://'):
        func = import_function(url)
        if image:
            func.spec.image = image
    elif url.endswith('.ipynb'):
        func = code_to_function(name, filename=url, image=image, kind=kind)
    elif url.endswith('.py'):
        if not image:
            raise ValueError('image must be provided with py code files, '
                             'use function object for more control/settings')
        if in_context and with_repo:
            func = new_function(name, command=url, image=image, kind=kind or 'job')
        else:
            func = code_to_function(name, filename=url, image=image, kind=kind or 'job')
    else:
        raise ValueError('unsupported function url {} or no spec'.format(url))

    if with_repo:
        func.spec.build.source = './'

    return _init_function_from_obj(func, project, name)


def _init_function_from_obj(func, project, name=None):
    build = func.spec.build
    if project.origin_url:
        origin = project.origin_url
        try:
            if project.repo:
                origin += '#' + project.repo.head.commit.hexsha
        except Exception:
            pass
        build.code_origin = origin
    if project.name:
        func.metadata.project = project.name
    if project.tag:
        func.metadata.tag = project.tag
    return name or func.metadata.name, func


def _create_pipeline(project, pipeline, funcs, secrets=None):
    functions = {}
    for name, func in funcs.items():
        f = func.copy()
        src = f.spec.build.source
        if project.source and src and src in ['.', './']:
            if project.mountdir:
                f.spec.workdir = project.mountdir
                f.spec.build.source = ''
            else:
                f.spec.build.source = project.source

        functions[name] = f

    spec = imputil.spec_from_file_location('workflow', pipeline)
    if spec is None:
        raise ImportError('cannot import workflow {}'.format(pipeline))
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    setattr(mod, 'funcs', functions)
    setattr(mod, 'this_project', project)

    if hasattr(mod, 'init_functions'):
        getattr(mod, 'init_functions')(functions, project, secrets)

    # verify all functions are in this project
    for f in functions.values():
        f.metadata.project = project.name

    if not hasattr(mod, 'kfpipeline'):
        raise ValueError('pipeline function (kfpipeline) not found')

    kfpipeline = getattr(mod, 'kfpipeline')
    return kfpipeline


def _run_pipeline(project, name, pipeline, functions, secrets=None,
                  arguments=None, artifact_path=None, namespace=None,
                  ttl=None):
    kfpipeline = _create_pipeline(project, pipeline, functions, secrets)

    namespace = namespace or config.namespace
    id = run_pipeline(kfpipeline, project=project.name, arguments=arguments,
                      experiment=name, namespace=namespace,
                      artifact_path=artifact_path, ttl=ttl)
    return id


def github_webhook(request):
    signature = request.headers.get('X-Hub-Signature')
    data = request.data
    print('sig:', signature)
    print('headers:', request.headers)
    print('data:', data)
    print('json:', request.get_json())

    if request.headers.get('X-GitHub-Event') == "ping":
        return {'msg': 'Ok'}

    return {'msg': 'pushed'}


def clone_git(url, context, secrets, clone):
    urlobj = urlparse(url)
    scheme = urlobj.scheme.lower()
    if not context:
        raise ValueError('please specify a target (context) directory for clone')

    if path.exists(context) and path.isdir(context):
        if clone:
            shutil.rmtree(context)
        else:
            try:
                repo = Repo(context)
                return _get_repo_url(repo), repo
            except Exception:
                pass

    host = urlobj.hostname or 'github.com'
    if urlobj.port:
        host += ':{}'.format(urlobj.port)

    token = urlobj.username or secrets.get('GITHUB_TOKEN') \
            or secrets.get('git_user')
    password = urlobj.password or secrets.get('git_password') \
               or 'x-oauth-basic'
    if token:
        clone_path = 'https://{}:{}@{}{}'.format(
            token, password, host, urlobj.path)
    else:
        clone_path = 'https://{}{}'.format(host, urlobj.path)

    branch = None
    if urlobj.fragment:
        refs = urlobj.fragment
        if refs.startswith('refs/'):
            branch = refs[refs.rfind('/')+1:]
        else:
            url = url.replace('#' + refs, '#refs/heads/{}'.format(refs))

    repo = Repo.clone_from(clone_path, context, single_branch=True, b=branch)
    return url, repo


def clone_tgz(url, context, secrets):
    if not context:
        raise ValueError('please specify a target (context) directory for clone')

    if path.exists(context) and path.isdir(context):
        shutil.rmtree(context)
    tmp = mktemp()
    download_object(url, tmp, secrets=secrets)
    tf = tarfile.open(tmp)
    tf.extractall(context)
    tf.close()
    remove(tmp)


def _get_repo_url(repo):
    url = ''
    remotes = [remote.url for remote in repo.remotes]
    if not remotes:
        return ''

    url = remotes[0]
    url = url.replace('https://', 'git://')
    try:
        url = '{}#refs/heads/{}'.format(url, repo.active_branch.name)
    except Exception:
        pass

    return url





