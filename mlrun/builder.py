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
import os
import tarfile
from base64 import b64encode
from tempfile import mktemp

from mlrun.k8s_utils import k8s_helper, BasePod
from mlrun.datastore import StoreManager

default_image = 'python:3.6-jessie'
mlrun_package = 'git+https://github.com/v3io/mlrun.git@development'
k8s = None


def make_dockerfile(base_image=default_image,
                    commands=None, src_dir=None,
                    requirements=None):
    dock = f'FROM {base_image}\n'
    dock += 'WORKDIR /run\n'
    if commands:
        dock += ''.join([f'RUN {b}\n' for b in commands])
    dock += f'RUN pip install {mlrun_package}\n'
    if src_dir:
        dock += f'ADD {src_dir} /run\n'
    if requirements:
        dock += f'RUN pip install -r {requirements}\n'

    print(dock)
    return dock


def make_kaniko_pod(context, dest,
                    dockerfile=None,
                    dockertext=None,
                    secret_name=None):

    if not dockertext and not dockerfile:
        raise ValueError('docker file or text must be specified')
    if not secret_name:
        raise ValueError('registry secret name must be specified')

    if dockertext:
        dockerfile = '/empty/Dockerfile'

    kpod=BasePod('kaniko',
                 'gcr.io/kaniko-project/executor:latest',
                 args=["--dockerfile", dockerfile,
                       "--context", context,
                       "--destination", dest])

    items = [{'key': '.dockerconfigjson', 'path': '.docker/config.json'}]
    kpod.mount_secret(secret_name, '/root/', items=items)

    kpod.mount_v3io(user='admin')
    if dockertext:
        kpod.mount_empty()
        kpod.set_init_container('alpine',
                                args=['sh', '-c', 'echo ${TEXT} | base64 -d > /empty/Dockerfile'],
                                env={'TEXT': dockertext})

    return kpod


def upload_tarfile(source_dir, target, secrets=None):
    tmpfile = mktemp('.tar.gz')
    with tarfile.open(tmpfile, "w:gz") as tar:
        tar.add(source_dir, arcname='')

    stores = StoreManager(secrets)
    datastore, subpath = stores.get_or_create_store(target)
    datastore.upload(subpath, tmpfile)


def build(dest,
          context,
          commands=None,
          source_dir=None,
          target='',
          base_image=default_image,
          requirements=None,
          secret_name='my-docker',
          secrets={},
          interactive=True):

    upload_tarfile(source_dir, target, secrets)

    global k8s
    if not k8s:
        k8s = k8s_helper('default-tenant', config_file='./config')

    src_file = os.path.basename(target)
    print(src_file)
    dock = make_dockerfile(base_image, commands, src_file, requirements)
    dock64 = b64encode(dock.encode('utf-8')).decode('utf-8')

    kpod = make_kaniko_pod(context, dest, dockertext=dock64, secret_name=secret_name)
    k8s.run_job(kpod)


build('yhaviv/ktests2:latest',
      context='/User/context',
      source_dir='./buildtst',
      target='v3ios:///users/admin/context/src.tar.gz',
      requirements='requirements.txt')

