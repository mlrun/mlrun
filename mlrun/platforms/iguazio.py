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
import json
import os
import requests
import urllib3
from datetime import datetime
from mlrun.config import config


_cached_control_session = None


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


def mount_v3io(name='v3io', remote='~/', mount_path='/User',
               access_key='', user='', secret=None):
    """
        Modifier function to apply to a Container Op to volume mount a v3io path
        Usage:
            train = train_op(...)
            train.apply(mount_v3io(container='users', sub_path='/iguazio', mount_path='/data'))
    """

    def _mount_v3io(task):
        from kubernetes import client as k8s_client
        vol = v3io_to_vol(name, remote, access_key, user, secret=secret)
        task.add_volume(vol).add_volume_mount(k8s_client.V1VolumeMount(mount_path=mount_path, name=name))

        if not secret:
            task = v3io_cred(access_key=access_key)(task)
        return (task)

    return _mount_v3io


def mount_spark_conf():
    def _mount_spark(task):
        from kubernetes import client as k8s_client
        task.add_volume_mount(k8s_client.V1VolumeMount(name='spark-master-config', mount_path='/etc/config/spark'))
        return (task)

    return _mount_spark


def mount_v3iod(namespace, v3io_config_configmap, v3io_auth_secret='spark-operator-v3io-auth'):
    def _mount_v3iod(task):
        from kubernetes import client as k8s_client

        def add_vol(name, mount_path, host_path):
            vol = k8s_client.V1Volume(name=name, host_path=k8s_client.V1HostPathVolumeSource(path=host_path, type=''))
            task.add_volume(vol).add_volume_mount(k8s_client.V1VolumeMount(mount_path=mount_path, name=name))

        add_vol(name='shm', mount_path = '/dev/shm', host_path='/dev/shm/' + namespace)
        add_vol(name='v3iod-comm', mount_path='/var/run/iguazio/dayman', host_path='/var/run/iguazio/dayman/' + namespace)

        vol = k8s_client.V1Volume(name='daemon-health', empty_dir=k8s_client.V1EmptyDirVolumeSource())
        task.add_volume(vol).add_volume_mount(k8s_client.V1VolumeMount(mount_path='/var/run/iguazio/daemon_health', name='daemon-health'))

        vol = k8s_client.V1Volume(name='v3io-config', config_map=k8s_client.V1ConfigMapVolumeSource(name=v3io_config_configmap, default_mode=420))
        task.add_volume(vol).add_volume_mount(k8s_client.V1VolumeMount(mount_path='/etc/config/v3io', name='v3io-config'))

        # vol = k8s_client.V1Volume(name='v3io-auth', secret=k8s_client.V1SecretVolumeSource(secret_name= v3io_auth_secret, default_mode= 420))
        # task.add_volume(vol).add_volume_mount(k8s_client.V1VolumeMount(mount_path='/igz/.igz', name='v3io-auth'))

        task.add_env_variable(k8s_client.V1EnvVar(name='CURRENT_NODE_IP', value_from=k8s_client.V1EnvVarSource(
        field_ref=k8s_client.V1ObjectFieldSelector(api_version='v1', field_path='status.hostIP'))))
        task.add_env_variable(k8s_client.V1EnvVar(name='IGZ_DATA_CONFIG_FILE', value='/igz/java/conf/v3io.conf'))

        return (task)

    return _mount_v3iod


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
        _user = user or environ.get('V3IO_USERNAME')
        _access_key = access_key or environ.get('V3IO_ACCESS_KEY')

        return (
            task
                .add_env_variable(k8s_client.V1EnvVar(name='V3IO_API', value=web_api))
                .add_env_variable(k8s_client.V1EnvVar(name='V3IO_USERNAME', value=_user))
                .add_env_variable(k8s_client.V1EnvVar(name='V3IO_ACCESS_KEY', value=_access_key))
        )

    return _use_v3io_cred


def split_path(mntpath=''):
    if mntpath[0] == '/':
        mntpath = mntpath[1:]
    paths = mntpath.split('/')
    container = paths[0]
    subpath = ''
    if len(paths) > 1:
        subpath = mntpath[len(container):]
    return container, subpath


def v3io_to_vol(name, remote='~/', access_key='', user='', secret=None):
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
    if secret:
        secret = {'name': secret}

    opts = {'accessKey': access_key, 'container': container, 'subPath': subpath}
    # vol = client.V1Volume(name=name, flex_volume=client.V1FlexVolumeSource('v3io/fuse', options=opts))
    vol = {'flexVolume': client.V1FlexVolumeSource(
        'v3io/fuse', options=opts, secret_ref=secret), 'name': name}
    return vol


class OutputStream:
    def __init__(self, stream_path, shards=1):
        import v3io
        self._v3io_client = v3io.dataplane.Client()
        self._container, self._stream_path = split_path(stream_path)
        response = self._v3io_client.create_stream(
            container=self._container,
            path=self._stream_path,
            shard_count=shards,
            raise_for_status=v3io.dataplane.RaiseForStatus.never)
        if not (response.status_code == 400 and 'ResourceInUse' in str(response.body)):
            response.raise_for_status([409, 204])

    def push(self, data):
        if not isinstance(data, list):
            data = [data]
        records = [{'data': json.dumps(rec)} for rec in data]
        response = self._v3io_client.put_records(container=self._container,
                                                 path=self._stream_path,
                                                 records=records)


def create_control_session(url, username, password):
    # for systems without production cert - silence no cert verification WARN
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    if not username or not password:
        raise ValueError('cannot create session key, missing username or password')

    session = requests.Session()
    session.auth = (username, password)
    try:
        auth = session.post(f'{url}/api/sessions', verify=False)
    except OSError as e:
        raise OSError('error: cannot connect to {}: {}'.format(url, e))

    if not auth.ok:
        raise OSError('failed to create session: {}, {}'.format(url, auth.text))

    return auth.json()['data']['id']


def is_iguazio_endpoint(endpoint_url: str) -> bool:
    # TODO: find a better heuristic
    return '.default-tenant.' in endpoint_url


def is_iguazio_control_session(value: str) -> bool:
    # TODO: find a better heuristic
    return len(value) > 20 and '-' in value


# we assign the control session to the password since this is iguazio auth scheme
# (requests should be sent with username:control_session as auth header)
def add_or_refresh_credentials(
        api_url: str, username: str = '', password: str = '',
        token: str = '') -> (str, str):

    # this may be called in "open source scenario" so in this case (not iguazio endpoint) simply do nothing
    if not is_iguazio_endpoint(api_url) or is_iguazio_control_session(password):
        return username, password, token

    username = username or os.environ.get('V3IO_USERNAME')
    password = password or os.environ.get('V3IO_PASSWORD')

    if not username or not password:
        raise ValueError('username and password needed to create session')

    global _cached_control_session
    now = datetime.now()
    if _cached_control_session:
        if _cached_control_session[2] == username and _cached_control_session[3] == password \
                and (now - _cached_control_session[1]).seconds < 20 * 60 * 60:
            return _cached_control_session[2], _cached_control_session[0], ''

    iguazio_dashboard_url = 'https://dashboard' + api_url[api_url.find('.'):]
    control_session = create_control_session(iguazio_dashboard_url, username, password)
    _cached_control_session = (control_session, now, username, password)
    return username, control_session, ''


