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

from ..model import ModelObj
import tarfile
from tempfile import mktemp
from git import Repo

import yaml
from os import path, remove

from ..datastore import download_object
from ..config import config
from ..run import import_function, code_to_function, new_function, run_pipeline
import importlib.util as imputil
from urllib.parse import urlparse
from kfp import compiler

from ..utils import update_in, new_pipe_meta, logger
from ..runtimes.utils import add_code_metadata


class ProjectError(Exception):
    pass


def new_project(name, context=None, functions=None, workflows=None,
                init_git=False):
    """Create a new MLRun project"""
    project = MlrunProject(name=name,
                           functions=functions,
                           workflows=workflows)
    project.context = context

    if init_git:
        repo = Repo.init(context)
        project.repo = repo

    return project


def load_project(context, url=None, name=None, secrets=None,
                 mount_url=None, init_git=False, subpath='', clone=False):
    """Load an MLRun project from git or tar or dir"""

    secrets = secrets or {}
    repo = None
    if url:
        if url.startswith('git://'):
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

    project = _load_project_dir(context, name, subpath)
    project.source = mount_url or url
    project.repo = repo
    if repo:
        project.branch = repo.active_branch.name
    project.origin_url = url
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
                                           'name': func.metadata.name}],
                               workflows=[])
    else:
        raise ValueError('project or function YAML not found in path')

    project.context = context
    project.name = name or project.name
    project.subpath = subpath
    return project


class MlrunProject(ModelObj):
    kind = 'project'

    def __init__(self, name=None, description=None, params=None,
                 functions=None, workflows=None, conda=None):

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
        self.workflows = workflows or []
        self._workflows = {}
        self._secrets = {}
        self.params = params or {}
        self.conda = conda or {}
        self.remote = False

        self._function_objects = {}
        self._function_defs = {}
        self.functions = functions or []

    @property
    def source(self) -> str:
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

    @property
    def functions(self) -> list:
        funcs = []
        for name, f in self._function_defs.items():
            if hasattr(f, 'to_dict'):
                spec = f.to_dict(strip=True)
                if f.spec.build.source.startswith(self._source_repo()):
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
            if not name or 'code' not in w:
                raise ValueError('workflow "name" and "code" must be specified')
            wfdict[name] = w

        self._workflows = wfdict

    def set_workflow(self, name, code):
        self._workflows[name] = {'name': name, 'code':code}

    def reload(self, sync=False):
        project = _load_project_dir(self.context, self.name, self.subpath)
        project.source = self.source
        project.repo = self.repo
        project.branch = self.branch
        project.origin_url = self.origin_url
        if sync:
            project.sync_functions()
        return project

    def set_function(self, func, name='', kind='', image=None):
        if isinstance(func, str):
            if not name:
                raise ValueError('function name must be specified')
            fdict = {'url': func, 'name': name, 'kind': kind, 'image': image}
            func = {k: v for k, v in fdict.items() if v}
            name, f = init_function_from_dict(func, self)
        elif hasattr(func, 'to_dict'):
            name, f = init_function_from_obj(func, self, name=name)
            if not name:
                raise ValueError('function name must be specified')
        else:
            raise ValueError('func must be a function url or object')

        self._function_defs[name] = func
        self._function_objects[name] = f
        return self

    def func(self, key, sync=False):
        if key not in self._function_defs:
            raise KeyError('function {} not found'.format(key))
        if sync or not self._initialized or key not in self._function_objects:
            self.sync_functions()
        return self._function_objects[key]

    def pull(self, branch=None, remote=None):
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
                name, func = init_function_from_obj(f, self)
            else:
                if not isinstance(f, dict):
                    raise ValueError('function must be an object or dict')
                name, func = init_function_from_dict(f, self)
            func.spec.build.code_origin = origin
            funcs[name] = func
            if save:
                func.save(versioned=False)

        self._function_objects = funcs
        self._initialized = True

    def with_secrets(self, secrets):
        self._secrets = secrets
        return self

    def run(self, name=None, workflow_path=None, arguments=None,
            artifact_path=None, namespace=None, sync=False, dirty=False):

        if self.repo and self.repo.is_dirty():
            msg = 'you seem to have uncommitted git changes, use .push()'
            if dirty:
                logger.warning('WARNING!, ' + msg)
            else:
                raise ProjectError(msg + ' or dirty=True')

        if self.repo and not self.source:
            raise ProjectError(
                'remote repo is not defined, use .create_remote() + push()')

        self.sync_functions(always=sync)
        if not self._function_objects:
            raise ValueError('no functions in the project')

        if not name and not workflow_path:
            raise ValueError('workflow name or path not specified')
        if not workflow_path:
            if name not in self._workflows:
                raise ValueError('workflow {} not found'.format(name))
            workflow_path = self._workflows.get(name)['code']
            workflow_path = path.join(self.context, workflow_path)

        name = '{}-{}'.format(self.name, name) if name else self.name
        run = _run_pipeline(name, workflow_path, self._function_objects,
                            self.params, secrets=self._secrets,
                            arguments=arguments, artifact_path=artifact_path,
                            namespace=namespace, remote=self.remote)
        return run

    def save_workflow(self, name, target, artifact_path=None):
        if not name or name not in self._workflows:
            raise ValueError('workflow {} not found'.format(name))

        wfpath = path.join(self.context, self._workflows.get(name)['code'])
        pipeline = create_pipeline(wfpath, self._function_objects,
                                   self.params, secrets=self._secrets,
                                   artifact_path=artifact_path)

        conf = new_pipe_meta(artifact_path)
        compiler.Compiler().compile(pipeline, target, pipeline_conf=conf)

    def clear_context(self):
        if self.context and path.exists(self.context) and path.isdir(self.context):
            shutil.rmtree(self.context)

    def save(self, filepath=None):
        filepath = filepath or path.join(self.context, self.subpath,
                                         'project.yaml')
        with open(filepath, 'w') as fp:
            fp.write(self.to_yaml())


def init_function_from_dict(f, project):
    name = f.get('name', '')
    url = f.get('url', '')
    kind = f.get('kind', '')
    image = f.get('image', None)

    in_context = False
    if not url and 'spec' not in f:
        raise ValueError('function missing a url or a spec')

    if url and '://' not in url:
        if project.context and not url.startswith('/'):
            url = path.join(project.context, url)
            in_context = True
        if not path.isfile(url):
            raise Exception('{} not found'.format(url))

    if 'spec' in f:
        func = new_function(name, runtime=f['spec'])
    elif url.endswith('.yaml') or url.startswith('db://'):
        func = import_function(url)
    elif url.endswith('.ipynb'):
        func = code_to_function(name, filename=url, image=image, kind=kind)
    elif url.endswith('.py'):
        if not image:
            raise ValueError('image must be provided with py code files')
        if in_context:
            func = new_function(name, command=url, image=image, kind=kind or 'job')
        else:
            func = code_to_function(name, filename=url, image=image, kind=kind or 'job')
    else:
        raise ValueError('unsupported function url {} or no spec'.format(url))

    return init_function_from_obj(func, project, name, in_context)


def init_function_from_obj(func, project, name=None, in_context=True):
    build = func.spec.build
    if project.source and in_context and \
            (not build.source or build.source in ['.', './']):
        build.source = project.source
        if project.repo:
            origin = project.origin_url
            try:
                origin += '#' + project.repo.head.commit.hexsha
            except Exception:
                pass
            build.code_origin = origin
    if project.name:
        func.metadata.project = project.name
    if project.tag:
        func.metadata.tag = project.tag

    return name or func.metadata.name, func


def create_pipeline(pipeline, functions, params=None, secrets=None,
                    artifact_path=None):

    spec = imputil.spec_from_file_location('workflow', pipeline)
    if spec is None:
        raise ImportError('cannot import workflow {}'.format(pipeline))
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    setattr(mod, 'artifact_path', artifact_path)
    setattr(mod, 'funcs', functions)

    if hasattr(mod, 'init_functions'):
        getattr(mod, 'init_functions')(functions, params, secrets)

    if not hasattr(mod, 'kfpipeline'):
        raise ValueError('pipeline function (kfpipeline) not found')

    kfpipeline = getattr(mod, 'kfpipeline')
    return kfpipeline


def _run_pipeline(name, pipeline, functions, params=None, secrets=None,
                  arguments=None, artifact_path=None, namespace=None,
                  remote=False):
    kfpipeline = create_pipeline(pipeline, functions, params, secrets,
                                 artifact_path)

    namespace = namespace or config.namespace
    id = run_pipeline(kfpipeline, arguments=arguments, experiment=name,
                      namespace=namespace, artifact_path=artifact_path,
                      remote=remote)
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

    token = urlobj.username or secrets.get('git_token') \
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
            branch = branch[branch.rfind('/')+1:]
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
        return '', ''

    url = remotes[0]
    url = url.replace('https://', 'git://')
    try:
        url = '{}#refs/heads/{}'.format(url, repo.active_branch.name)
    except Exception:
        pass

    return url





