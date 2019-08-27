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

from os import environ
import tarfile
from base64 import b64encode
from tempfile import mktemp

from mlrun.k8s_utils import k8s_helper, BasePod
from mlrun.datastore import StoreManager

default_image = 'python:3.6-jessie'
mlrun_package = 'git+https://github.com/v3io/mlrun.git'
kaniko_version = 'v0.9.0'
k8s = None


def make_dockerfile(base_image=default_image,
                    commands=None, src_dir=None,
                    requirements=None):
    dock = f'FROM {base_image}\n'
    dock += 'WORKDIR /run\n'
    if src_dir:
        dock += f'ADD {src_dir} /run\n'
    if requirements:
        #dock += f'ADD {requirements} /run\n'
        dock += f'RUN pip install -r {requirements}\n'
    if commands:
        dock += ''.join([f'RUN {b}\n' for b in commands])
    dock += 'ENV PYTHONPATH /run'

    print(dock)
    return dock


def make_kaniko_pod(context, dest,
                    dockerfile=None,
                    dockertext=None,
                    inline_code=None,
                    requirements=None,
                    secret_name=None,
                    verbose=False):

    if not dockertext and not dockerfile:
        raise ValueError('docker file or text must be specified')

    if dockertext:
        dockerfile = '/empty/Dockerfile'

    args = ["--dockerfile", dockerfile,
            "--context", context,
            "--destination", dest]
    if not secret_name:
        args.append('--insecure')
    if verbose:
        args += ["--verbosity", 'debug']

    kpod=BasePod('mlrun-build',
                 'gcr.io/kaniko-project/executor:' + kaniko_version,
                 args=args,
                 kind='build')

    if secret_name:
        items = [{'key': '.dockerconfigjson', 'path': '.docker/config.json'}]
        kpod.mount_secret(secret_name, '/root/', items=items)

    if dockertext or inline_code or requirements:
        kpod.mount_empty()
        commands = []
        env = {}
        if dockertext:
            commands.append('echo ${DOCKERFILE} | base64 -d > /empty/Dockerfile')
            env['DOCKERFILE'] = b64encode(dockertext.encode('utf-8')).decode('utf-8')
        if inline_code:
            commands.append('echo ${CODE} | base64 -d > /empty/main.py')
            env['CODE'] = b64encode(inline_code.encode('utf-8')).decode('utf-8')
        if requirements:
            commands.append('echo ${REQUIREMENTS} | base64 -d > /empty/requirements.txt')
            env['REQUIREMENTS'] = b64encode('\n'.join(requirements).encode('utf-8')).decode('utf-8')

        kpod.set_init_container('alpine', args=['sh', '-c', '; '.join(commands)], env=env)

    return kpod


def upload_tarball(source_dir, target, secrets=None):
    tmpfile = mktemp('.tar.gz')
    with tarfile.open(tmpfile, "w:gz") as tar:
        tar.add(source_dir, arcname='')

    stores = StoreManager(secrets)
    datastore, subpath = stores.get_or_create_store(target)
    datastore.upload(subpath, tmpfile)


def build_image(dest,
                commands=None,
                source='',
                mounter='v3io',
                base_image=None,
                requirements=None,
                inline_code=None,
                secret_name=None,
                namespace=None,
                with_mlrun=True,
                registry=None,
                interactive=True,
                verbose=False):

    global k8s
    if not k8s:
        k8s = k8s_helper(namespace or 'default-tenant')

    if registry:
        dest = '{}/{}'.format(registry, dest)
    elif 'DOCKER_REGISTRY_SERVICE_HOST' in environ:
        dest = '{}:5000/{}'.format(environ.get('DOCKER_REGISTRY_SERVICE_HOST'), dest)

    if isinstance(requirements, list):
        requirements_list = requirements
        requirements_path = 'requirements.txt'
        if source:
            raise ValueError('requirements list only works with inline code')
    else:
        requirements_list = None
        requirements_path = requirements

    base_image = base_image or default_image
    if with_mlrun:
        commands = commands or []
        commands.append(f'pip install {mlrun_package}')
    context = '/context'
    to_mount = False
    src_dir = '.'
    if inline_code:
        context = '/empty'
    elif '://' in source:
        context = source
    elif source:
        to_mount = True
        if source.endswith('.tar.gz'):
            source, src_dir = os.path.split(source)
    else:
        src_dir = None

    dock = make_dockerfile(base_image, commands,
                           src_dir=src_dir,
                           requirements=requirements_path)

    kpod = make_kaniko_pod(context, dest, dockertext=dock,
                           inline_code=inline_code, requirements=requirements_list,
                           secret_name=secret_name, verbose=verbose)

    if to_mount:
        # todo: support different mounters
        kpod.mount_v3io(remote=source, mount_path='/context')

    if interactive:
        return k8s.run_job(kpod)
    else:
        return k8s.create_pod(kpod)
