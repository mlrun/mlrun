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
from http import HTTPStatus
from datetime import datetime
from collections import namedtuple


_cached_control_session = None


def xcp_op(
    src, dst, f="", recursive=False, mtime="", log_level="info", minsize=0, maxsize=0
):
    """Parallel cloud copy."""
    from kfp import dsl

    args = [
        # '-f', f,
        # '-t', mtime,
        # '-m', maxsize,
        # '-n', minsize,
        # '-v', log_level,
        src,
        dst,
    ]
    if recursive:
        args = ["-r"] + args

    return dsl.ContainerOp(
        name="xcp", image="yhaviv/invoke", command=["xcp"], arguments=args,
    )


Mount = namedtuple("Mount", ["path", "sub_path"])


def mount_v3io_extended(
    name="v3io", remote="", mounts=None, access_key="", user="", secret=None
):
    """Modifier function to apply to a Container Op to volume mount a v3io path

    :param name:            the volume name
    :param remote:          the v3io path to use for the volume. ~/ prefix will be replaced with /users/<username>/
    :param mounts:          list of mount & volume sub paths (type Mount). empty mounts & remote mount /v3io & /User
    :param access_key:      the access key used to auth against v3io. if not given V3IO_ACCESS_KEY env var will be used
    :param user:            the username used to auth against v3io. if not given V3IO_USERNAME env var will be used
    :param secret:          k8s secret name which would be used to get the username and access key to auth against v3io.
    """

    # Empty remote & mounts defaults are mounts of /v3io and /User
    if not remote and not mounts:
        user = os.environ.get("V3IO_USERNAME", user)
        if not user:
            raise ValueError(
                "user name/env must be specified when using empty remote and mounts"
            )
        mounts = [
            Mount(path="/v3io", sub_path=""),
            # Temporarily commented out as we do not support multiple mount on the same volume yet (set_named_item...)
            #            Mount(path="/User", sub_path="users/" + user),
        ]

    if not isinstance(mounts, list) and any([not isinstance(x, Mount) for x in mounts]):
        raise TypeError("mounts should be a list of Mount")

    def _mount_v3io_extended(task):
        from kubernetes import client as k8s_client

        vol = v3io_to_vol(name, remote, access_key, user, secret=secret)
        task.add_volume(vol)
        for mount in mounts:
            task.add_volume_mount(
                k8s_client.V1VolumeMount(
                    mount_path=mount.path, sub_path=mount.sub_path, name=name
                )
            )

        if not secret:
            task = v3io_cred(access_key=access_key)(task)
        return task

    return _mount_v3io_extended


def mount_v3io(
    name="v3io", remote="~/", mount_path="/User", access_key="", user="", secret=None
):
    """Modifier function to apply to a Container Op to volume mount a v3io path

    :param name:            the volume name
    :param remote:          the v3io path to use for the volume. ~/ prefix will be replaced with /users/<username>/
    :param mount_path:      the volume mount path
    :param access_key:      the access key used to auth against v3io. if not given V3IO_ACCESS_KEY env var will be used
    :param user:            the username used to auth against v3io. if not given V3IO_USERNAME env var will be used
    :param secret:          k8s secret name which would be used to get the username and access key to auth against v3io.
    """

    return mount_v3io_extended(
        name=name,
        remote=remote,
        mounts=[Mount(path=mount_path, sub_path="")],
        access_key=access_key,
        user=user,
        secret=secret,
    )


def mount_spark_conf():
    def _mount_spark(task):
        from kubernetes import client as k8s_client

        task.add_volume_mount(
            k8s_client.V1VolumeMount(
                name="spark-master-config", mount_path="/etc/config/spark"
            )
        )
        return task

    return _mount_spark


def mount_v3iod(namespace, v3io_config_configmap):
    def _mount_v3iod(task):
        from kubernetes import client as k8s_client

        def add_vol(name, mount_path, host_path):
            vol = k8s_client.V1Volume(
                name=name,
                host_path=k8s_client.V1HostPathVolumeSource(path=host_path, type=""),
            )
            task.add_volume(vol).add_volume_mount(
                k8s_client.V1VolumeMount(mount_path=mount_path, name=name)
            )

        add_vol(name="shm", mount_path="/dev/shm", host_path="/dev/shm/" + namespace)
        add_vol(
            name="v3iod-comm",
            mount_path="/var/run/iguazio/dayman",
            host_path="/var/run/iguazio/dayman/" + namespace,
        )

        vol = k8s_client.V1Volume(
            name="daemon-health", empty_dir=k8s_client.V1EmptyDirVolumeSource()
        )
        task.add_volume(vol).add_volume_mount(
            k8s_client.V1VolumeMount(
                mount_path="/var/run/iguazio/daemon_health", name="daemon-health"
            )
        )

        vol = k8s_client.V1Volume(
            name="v3io-config",
            config_map=k8s_client.V1ConfigMapVolumeSource(
                name=v3io_config_configmap, default_mode=420
            ),
        )
        task.add_volume(vol).add_volume_mount(
            k8s_client.V1VolumeMount(mount_path="/etc/config/v3io", name="v3io-config")
        )

        task.add_env_variable(
            k8s_client.V1EnvVar(
                name="CURRENT_NODE_IP",
                value_from=k8s_client.V1EnvVarSource(
                    field_ref=k8s_client.V1ObjectFieldSelector(
                        api_version="v1", field_path="status.hostIP"
                    )
                ),
            )
        )
        task.add_env_variable(
            k8s_client.V1EnvVar(
                name="IGZ_DATA_CONFIG_FILE", value="/igz/java/conf/v3io.conf"
            )
        )

        return task

    return _mount_v3iod


def v3io_cred(api="", user="", access_key=""):
    """
    Modifier function to copy local v3io env vars to task
    Usage:
        train = train_op(...)
        train.apply(use_v3io_cred())
    """

    def _use_v3io_cred(task):
        from kubernetes import client as k8s_client
        from os import environ

        web_api = api or environ.get("V3IO_API")
        _user = user or environ.get("V3IO_USERNAME")
        _access_key = access_key or environ.get("V3IO_ACCESS_KEY")

        return (
            task.add_env_variable(k8s_client.V1EnvVar(name="V3IO_API", value=web_api))
            .add_env_variable(k8s_client.V1EnvVar(name="V3IO_USERNAME", value=_user))
            .add_env_variable(
                k8s_client.V1EnvVar(name="V3IO_ACCESS_KEY", value=_access_key)
            )
        )

    return _use_v3io_cred


def split_path(mntpath=""):
    if mntpath[0] == "/":
        mntpath = mntpath[1:]
    paths = mntpath.split("/")
    container = paths[0]
    subpath = ""
    if len(paths) > 1:
        subpath = mntpath[len(container) :]
    return container, subpath


def v3io_to_vol(name, remote="~/", access_key="", user="", secret=None):
    from os import environ
    from kubernetes import client

    access_key = access_key or environ.get("V3IO_ACCESS_KEY")
    opts = {"accessKey": access_key}

    remote = str(remote)

    if remote.startswith("~/"):
        user = environ.get("V3IO_USERNAME", user)
        if not user:
            raise ValueError('user name/env must be specified when using "~" in path')
        if remote == "~/":
            remote = "users/" + user
        else:
            remote = "users/" + user + remote[1:]
    if remote:
        container, subpath = split_path(remote)
        opts["container"] = container
        opts["subPath"] = subpath

    if secret:
        secret = {"name": secret}

    # vol = client.V1Volume(name=name, flex_volume=client.V1FlexVolumeSource('v3io/fuse', options=opts))
    vol = {
        "flexVolume": client.V1FlexVolumeSource(
            "v3io/fuse", options=opts, secret_ref=secret
        ),
        "name": name,
    }
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
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
        )
        if not (response.status_code == 400 and "ResourceInUse" in str(response.body)):
            response.raise_for_status([409, 204])

    def push(self, data):
        if not isinstance(data, list):
            data = [data]
        records = [{"data": json.dumps(rec)} for rec in data]
        self._v3io_client.put_records(
            container=self._container, path=self._stream_path, records=records
        )


def create_control_session(url, username, password):
    # for systems without production cert - silence no cert verification WARN
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    if not username or not password:
        raise ValueError("cannot create session key, missing username or password")

    session = requests.Session()
    session.auth = (username, password)
    try:
        auth = session.post(f"{url}/api/sessions", verify=False)
    except OSError as e:
        raise OSError("error: cannot connect to {}: {}".format(url, e))

    if not auth.ok:
        raise OSError("failed to create session: {}, {}".format(url, auth.text))

    return auth.json()["data"]["id"]


def is_iguazio_endpoint(endpoint_url: str) -> bool:
    # TODO: find a better heuristic
    return ".default-tenant." in endpoint_url


def is_iguazio_control_session(value: str) -> bool:
    # TODO: find a better heuristic
    return len(value) > 20 and "-" in value


def is_iguazio_system_2_10_or_above(dashboard_url):
    # for systems without production cert - silence no cert verification WARN
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(f"{dashboard_url}/api/external_versions", verify=False)

    if not response.ok:
        if response.status_code == HTTPStatus.NOT_FOUND.value:
            # in iguazio systems prior to 2.10 this endpoint didn't exist, so the api returns 404 cause endpoint not
            # found
            return False
        response.raise_for_status()

    return True


# we assign the control session or access key to the password since this is iguazio auth scheme
# (requests should be sent with username:control_session/access_key as auth header)
def add_or_refresh_credentials(
    api_url: str, username: str = "", password: str = "", token: str = ""
) -> (str, str, str):

    # this may be called in "open source scenario" so in this case (not iguazio endpoint) simply do nothing
    if not is_iguazio_endpoint(api_url) or is_iguazio_control_session(password):
        return username, password, token

    username = username or os.environ.get("V3IO_USERNAME")
    password = password or os.environ.get("V3IO_PASSWORD")
    token = token or os.environ.get("V3IO_ACCESS_KEY")
    iguazio_dashboard_url = "https://dashboard" + api_url[api_url.find(".") :]

    # in 2.8 mlrun api is protected with control session, from 2.10 it's protected with access key
    is_access_key_auth = is_iguazio_system_2_10_or_above(iguazio_dashboard_url)
    if is_access_key_auth:
        if not username or not token:
            raise ValueError(
                "username and access key required to authenticate against iguazio system"
            )
        return username, token, ""

    if not username or not password:
        raise ValueError("username and password needed to create session")

    global _cached_control_session
    now = datetime.now()
    if _cached_control_session:
        if (
            _cached_control_session[2] == username
            and _cached_control_session[3] == password
            and (now - _cached_control_session[1]).seconds < 20 * 60 * 60
        ):
            return _cached_control_session[2], _cached_control_session[0], ""

    control_session = create_control_session(iguazio_dashboard_url, username, password)
    _cached_control_session = (control_session, now, username, password)
    return username, control_session, ""
