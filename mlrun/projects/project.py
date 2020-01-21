import shutil

from mlrun.model import ModelObj
import tarfile
from tempfile import mktemp

import yaml
from git import Repo
import json
from os import path, environ, remove

from mlrun.datastore import StoreManager
from mlrun import import_function, code_to_function, new_function
import importlib.util as imputil
from urllib.parse import urlparse
from kfp import dsl, Client


def load_project(context, url=None, secrets=None):

    secrets = secrets or {}
    source = workdir = None
    if url and url.startswith('git://'):
        source, workdir = clone_git(url, context, secrets)
    elif url and url.endswith('.tar.gz'):
        source, workdir = clone_tgz(url, context, secrets)

    fpath = path.join(context, 'project.yaml')
    if path.isfile(fpath):
        with open(fpath) as fp:
            data = fp.read()
            struct = yaml.load(data, Loader=yaml.FullLoader)
            project = MlrunProject.from_dict(struct)
            project.source = source

    elif path.isfile(path.join(context, 'function.yaml')):
        func = import_function(path.join(context, 'function.yaml'))
        project = MlrunProject(name=func.metadata.project,
                               context=context,
                               functions=[{'url': 'function.yaml'}],
                               workflows={})

    else:
        raise ValueError('project or function YAML not found in path')

    project.context = context
    return project


class MlrunProject(ModelObj):
    kind = 'project'

    def __init__(self, name=None, source=None, context=None,
                 functions=None, workflows=None):

        self._function_objects = {}
        self.name = name
        self.source = source
        self.context = context
        self.workflows = workflows or {}
        self.source = None
        self.config = {}
        self._secrets = {}
        self.params = {}

        self._functions = None
        self.functions = functions or []

    @property
    def functions(self):
        return self._functions

    @functions.setter
    def functions(self, funcs):
        self._functions = funcs
        self._init_funcs()

    def _init_funcs(self):
        self._function_objects = init_functions(self)

    def with_secrets(self, secrets):
        self._secrets = secrets
        return self

    def run(self, name=None, workflow_path=None,
            arguments=None, artifacts_path=None, namespace=None):
        if not self._function_objects:
            raise ValueError('no functions in the project')

        if not self.workflows and not workflow_path:
            raise ValueError('no workflows specified')
        if not workflow_path:
            if not name:
                name = list(self.workflows)[0]
            elif name not in self.workflows:
                raise ValueError('workflow {} not found'.format(name))
            workflow_path = self.workflows.get(name)

        name = '{}-{}'.format(self.name, name) if name else self.name
        wfpath = path.join(self.context, workflow_path)
        run = run_pipeline(name, wfpath, self._function_objects, self.params,
                           secrets=self._secrets, arguments=arguments,
                           artifacts_path=artifacts_path, namespace=namespace)
        return run

    def clear_context(self):
        if self.context and path.exists(self.context) and path.isdir(self.context):
            shutil.rmtree(self.context)


def init_functions(project):
    funcs = {}
    func_defs = project.functions
    for f in func_defs:
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
                raise Exception('function.yaml not found')

        if 'spec' in f:
            func = new_function(runtime=f['spec'])
        elif url.endswith('.yaml') or url.startswith('db://'):
            func = import_function(url)
        elif url.endswith('.ipynb'):
            func = code_to_function(filename=url, image=image, kind=kind)
        elif url.endswith('.py'):
            if not image:
                raise ValueError('image must be provided with py code files')
            func = code_to_function(filename=url, image=image, kind=kind or 'job')

        else:
            raise ValueError('unsupported function url {} or no spec'.format(url))

        name = name or func.metadata.name
        if project.source and in_context:
            func.spec.build.source = project.source
        funcs[name] = func

    return funcs


def run_pipeline(name, pipeline, functions, params=None, secrets=None,
                 arguments=None, artifacts_path=None, namespace=None):

    spec = imputil.spec_from_file_location('workflow', pipeline)
    if spec is None:
        raise ImportError('cannot import workflow {}'.format(pipeline))
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)

    setattr(mod, 'artifacts_path', artifacts_path)
    setattr(mod, 'funcs', functions)

    if hasattr(mod, 'init_functions'):
        getattr(mod, 'init_functions')(functions, params, secrets)

    if not hasattr(mod, 'kfpipeline'):
        raise ValueError('pipeline function (kfpipeline) not found')

    kfpipeline = getattr(mod, 'kfpipeline')
    client = Client(namespace=namespace or 'default-tenant')

    run_result = client.create_run_from_pipeline_func(
        kfpipeline, arguments, experiment_name=name)

    return run_result


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


def clone_git(url, context, config={}):
    urlobj = urlparse(url)
    scheme = urlobj.scheme.lower()

    host = urlobj.hostname or 'github.com'
    if urlobj.port:
        host += ':{}'.format(urlobj.port)

    token = urlobj.username or config.get('git_token')
    if token:
        clone_path = 'https://{}:x-oauth-basic@{}{}'.format(token, host, urlobj.path)
    else:
        clone_path = 'https://{}{}'.format(host, urlobj.path)

    workdir = None
    branch = None
    if urlobj.fragment:
        parts = urlobj.fragment.split(':')
        branch = parts[0]
        if branch.startswith('refs/'):
            branch = branch[branch.rfind('/')+1:]
        else:
            url = 'git://{}{}#refs/heads/{}'.format(host, urlobj.path, branch)
        if len(parts) > 1:
            workdir = parts[1]

    Repo.clone_from(clone_path, context, single_branch=True, b=branch)
    return url, workdir


def clone_tgz(url, context, config={}):
    stores = StoreManager(config.get('secrets', None))
    datastore, subpath = stores.get_or_create_store(url)
    tmp = mktemp()
    datastore.download(subpath, tmp)
    tf = tarfile.open(tmp)
    tf.extractall(context)
    tf.close()
    remove(tmp)

    return url, ''





