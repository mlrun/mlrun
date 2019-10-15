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

def xcp_op(src, dst, f='', recursive=False, mtime='', log_level='info', minsize=0, maxsize=0):
    """Parallel cloud copy."""
    from kfp import dsl
    args = [
        # '-f', f,
        # '-t', mtime,
        # '-m', maxsize,
        # '-n', minsize,
        # '-v', log_level,
        src, dst,
    ]
    if recursive:
        args = ['-r'] + args

    return dsl.ContainerOp(
        name='xcp',
        image='yhaviv/invoke',
        command=['xcp'],
        arguments=args,
    )


def mount_v3io(name='v3io', remote='~/', mount_path='/User', access_key='', user=''):
    """
        Modifier function to apply to a Container Op to volume mount a v3io path
        Usage:
            train = train_op(...)
            train.apply(mount_v3io(container='users', sub_path='/iguazio', mount_path='/data'))
    """

    def _mount_v3io(task):
        from kubernetes import client as k8s_client
        vol = v3io_to_vol(name, remote, access_key, user)
        task.add_volume(vol).add_volume_mount(k8s_client.V1VolumeMount(mount_path=mount_path, name=name))

        task = v3io_cred(access_key=access_key)(task)
        return (task)

    return _mount_v3io


def v3io_cred(api='', user='', access_key=''):
    """
        Modifier function to copy local v3io env vars to task
        Usage:
            train = train_op(...)
            train.apply(use_v3io_cred())
    """

    def _use_v3io_cred(task):
        from kubernetes import client as k8s_client
        from os import environ
        web_api = api or environ.get('V3IO_API')
        _user = environ.get('V3IO_USERNAME')
        _access_key = environ.get('V3IO_ACCESS_KEY')

        return (
            task
                .add_env_variable(k8s_client.V1EnvVar(name='V3IO_API', value=web_api))
                .add_env_variable(k8s_client.V1EnvVar(name='V3IO_USERNAME', value=_user))
                .add_env_variable(k8s_client.V1EnvVar(name='V3IO_ACCESS_KEY', value=_access_key))
        )

    return _use_v3io_cred


def add_env(env={}):
    """
        Modifier function to add env vars from dict
        Usage:
            train = train_op(...)
            train.apply(add_env({'MY_ENV':'123'}))
    """

    def _add_env(task):
        from kubernetes import client as k8s_client
        for k, v in env:
            task.add_env_variable(k8s_client.V1EnvVar(name=k, value=v))
        return task

    return _add_env


def split_path(mntpath=''):
    if mntpath[0] == '/':
        mntpath = mntpath[1:]
    paths = mntpath.split('/')
    container = paths[0]
    subpath = ''
    if len(paths) > 1:
        subpath = mntpath[len(container):]
    return container, subpath


def kaniko_op(image, context_path, secret_name='docker-secret'):
    """use kaniko to build Docker image."""

    from kubernetes import client as k8s_client
    cops = dsl.ContainerOp(
        name='kaniko',
        image='gcr.io/kaniko-project/executor:latest',
        arguments=["--dockerfile", "/context/Dockerfile",
                   "--context", "/context",
                   "--destination", image],
    )

    cops.add_volume(
        k8s_client.V1Volume(
            name='registry-creds',
            secret=k8s_client.V1SecretVolumeSource(
                secret_name=secret_name,
                items=[{'key': '.dockerconfigjson', 'path': '.docker/config.json'}],
            )
        ))
    cops.container.add_volume_mount(
        k8s_client.V1VolumeMount(
            name='registry-creds',
            mount_path='/root/',
        )
    )
    return cops.apply(mount_v3io(remote=context_path, mount_path='/context'))


def v3io_to_vol(name, remote='~/', access_key='', user=''):
    from os import environ
    from kubernetes import client
    access_key = access_key or environ.get('V3IO_ACCESS_KEY')
    remote = str(remote)

    if remote.startswith('~/'):
        user = environ.get('V3IO_USERNAME', user)
        if not user:
            raise ValueError('user name/env must be specified when using "~" in path')
        if remote == '~/':
            remote = 'users/' + user
        else:
            remote = 'users/' + user + remote[1:]
    container, subpath = split_path(remote)

    opts = {'accessKey': access_key, 'container': container, 'subPath': subpath}
    vol = client.V1Volume(name=name, flex_volume=client.V1FlexVolumeSource('v3io/fuse', options=opts))
    return vol
