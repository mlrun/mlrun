import shutil

from mlrun.model import ModelObj
import tarfile
from tempfile import mktemp

import yaml
from os import path, remove

from mlrun.datastore import StoreManager
from mlrun.config import config
from mlrun import import_function, code_to_function, new_function
import importlib.util as imputil
from urllib.parse import urlparse
from kfp import Client, compiler

from ..utils import update_in
from ..runtimes.utils import add_code_metadata


def new_project(name, context=None, functions=None, workflows=None):
    project = MlrunProject(name=name,
                           functions=functions,
                           workflows=workflows,
                           context=context)
    return project


def load_project(context, url=None, name=None, secrets=None,
                 mount_url=None, clone=True):

    secrets = secrets or {}
    source = workdir = branch = repo = None
    if url and url.startswith('git://'):
        source, workdir, branch, repo = clone_git(url, context, secrets, clone)
    elif url and url.endswith('.tar.gz'):
        source, workdir = clone_tgz(url, context, secrets, clone)

    fpath = path.join(context, 'project.yaml')
    if path.isfile(fpath):
        with open(fpath) as fp:
            data = fp.read()
            struct = yaml.load(data, Loader=yaml.FullLoader)
            struct['context'] = context
            struct['name'] = name or struct.get('name', '')
            project = MlrunProject.from_dict(struct)

    elif path.isfile(path.join(context, 'function.yaml')):
        func = import_function(path.join(context, 'function.yaml'))
        project = MlrunProject(name=func.metadata.project,
                               functions=[{'url': 'function.yaml',
                                           'name': func.metadata.name}],
                               workflows={})
    else:
        raise ValueError('project or function YAML not found in path')

    project.source = mount_url or source
    project.repo = repo
    project.branch = branch
    project.context = context
    project.origin_url = url
    project.name = name or project.name
    return project


class MlrunProject(ModelObj):
    kind = 'project'

    def __init__(self, name=None, source=None, params=None,
                 functions=None, workflows=None, tag=None):

        self._initialized = False
        self.name = name
        self.tag = tag
        self.origin_url = ''
        self.source = source
        self.context = None
        self.branch = None
        self.repo = None
        self.workflows = workflows or {}
        self._secrets = {}
        self.params = params or {}

        self._function_objects = {}
        self._function_defs = {}
        self.functions = functions or []

    @property
    def functions(self) -> list:
        funcs = []
        for name, f in self._function_defs.items():
            if hasattr(f, 'to_dict'):
                spec = f.to_dict()
                if f.spec.build.source == self.source:
                    update_in(spec, 'spec.build.source', './')
                # TODO: clean/change elements before persisting e.g. source
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

    def pull(self):
        url = self.origin_url
        if url and url.startswith('git://'):
            if not self.repo:
                raise ValueError('repo was not initialized, use load_project()')
            self.repo.git.pull()
        elif url and url.endswith('.tar.gz'):
            if not self.context:
                raise ValueError('target dit (context) is not set')
            clone_tgz(url, self.context, self._secrets, False)

    def sync_functions(self, always=True):
        if self._initialized and not always:
            return

        funcs = {}
        names = []
        origin = add_code_metadata(self.context)
        for name, f in self._function_defs.items():
            if hasattr(f, 'to_dict'):
                name, func = init_function_from_obj(f, self)
            else:
                if not isinstance(f, dict):
                    raise ValueError('function must be an object or dict')
                name, func = init_function_from_dict(f, self)
            func.spec.build.code_origin = origin
            funcs[name] = func
            names.append(name)

        self._function_objects = funcs
        self._initialized = True

    def with_secrets(self, secrets):
        self._secrets = secrets
        return self

    def run(self, name=None, workflow_path=None, arguments=None,
            artifacts_path=None, namespace=None, sync=False):

        self.sync_functions(sync)
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

    def save_workflow(self, name, target, artifacts_path=None):
        if not name or name not in self.workflows:
            raise ValueError('workflow {} not found'.format(name))

        wfpath = self.workflows.get(name)
        pipeline = create_pipeline(wfpath, self._function_objects,
                                   self.params, secrets=self._secrets,
                                   artifacts_path=artifacts_path)

        compiler.Compiler().compile(pipeline, target)

    def clear_context(self):
        if self.context and path.exists(self.context) and path.isdir(self.context):
            shutil.rmtree(self.context)


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

    return init_function_from_obj(func, project, name, in_context)


def init_function_from_obj(func, project, name=None, in_context=True):
    build = func.spec.build
    if project.source and in_context and \
            (not build.source or build.source in ['.', './']):
        build.source = project.source
    if project.name:
        func.metadata.project = project.name
    if project.tag:
        func.metadata.tag = project.tag

    return name or func.metadata.name, func


def create_pipeline(pipeline, functions, params=None, secrets=None,
                    artifacts_path=None):

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
    return kfpipeline


def run_pipeline(name, pipeline, functions, params=None, secrets=None,
                 arguments=None, artifacts_path=None, namespace=None):
    kfpipeline = create_pipeline(pipeline, functions, params, secrets,
                                 artifacts_path)
    client = Client(namespace=namespace or config.namespace)
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


def clone_git(url, context, secrets, clone=True):
    try:
        from git import Repo
    except ImportError as e:
        print('Failed to import git, pip install git')
        raise e

    urlobj = urlparse(url)
    scheme = urlobj.scheme.lower()
    if not context:
        raise ValueError('please specify a target (context) directory for clone')

    if clone and path.exists(context) and path.isdir(context):
        shutil.rmtree(context)

    host = urlobj.hostname or 'github.com'
    if urlobj.port:
        host += ':{}'.format(urlobj.port)

    token = urlobj.username or secrets.get('git_token')
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

    repo = Repo.clone_from(clone_path, context, single_branch=True, b=branch)
    return url, workdir, branch, repo


def clone_tgz(url, context, secrets, clone=True):
    if not context:
        raise ValueError('please specify a target (context) directory for clone')

    if path.exists(context) and path.isdir(context):
        shutil.rmtree(context)
    stores = StoreManager(secrets)
    datastore, subpath = stores.get_or_create_store(url)
    tmp = mktemp()
    datastore.download(subpath, tmp)
    tf = tarfile.open(tmp)
    tf.extractall(context)
    tf.close()
    remove(tmp)

    return url, ''





