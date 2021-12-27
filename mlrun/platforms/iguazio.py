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
import warnings
from collections import namedtuple
from datetime import datetime
from http import HTTPStatus
from urllib.parse import urlparse

import requests
import urllib3
import v3io

import mlrun.errors
from mlrun.config import config as mlconf

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


VolumeMount = namedtuple("Mount", ["path", "sub_path"])


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
    if remote and not mounts:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "mounts must be specified when remote is given"
        )

    # Empty remote & mounts defaults are mounts of /v3io and /User
    if not remote and not mounts:
        user = _resolve_mount_user(user)
        if not user:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "user name/env must be specified when using empty remote and mounts"
            )
        mounts = [
            VolumeMount(path="/v3io", sub_path=""),
            VolumeMount(path="/User", sub_path="users/" + user),
        ]

    if not isinstance(mounts, list) and any(
        [not isinstance(x, VolumeMount) for x in mounts]
    ):
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
            task = v3io_cred(access_key=access_key, user=user)(task)
        return task

    return _mount_v3io_extended


def _resolve_mount_user(user=None):
    return user or os.environ.get("V3IO_USERNAME")


def mount_v3io(
    name="v3io",
    remote="",
    mount_path="",
    access_key="",
    user="",
    secret=None,
    volume_mounts=None,
):
    """Modifier function to apply to a Container Op to volume mount a v3io path

    :param name:            the volume name
    :param remote:          the v3io path to use for the volume. ~/ prefix will be replaced with /users/<username>/
    :param mount_path:      the volume mount path (deprecated, exists for backwards compatibility, prefer to
                            use mounts instead)
    :param access_key:      the access key used to auth against v3io. if not given V3IO_ACCESS_KEY env var will be used
    :param user:            the username used to auth against v3io. if not given V3IO_USERNAME env var will be used
    :param secret:          k8s secret name which would be used to get the username and access key to auth against v3io.
    :param volume_mounts:   list of VolumeMount. empty volume mounts & remote will default to mount /v3io & /User.
    """
    if mount_path and volume_mounts:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "mount_path and mounts can not be given toegther"
        )

    if mount_path:
        warnings.warn(
            "mount_path is pending deprecation, use mounts instead"
            "This will be deprecated in 0.8.0, and will be removed in 0.10.0",
            # TODO: In 0.8.0 do changes in examples & demos In 0.10.0 remove
            PendingDeprecationWarning,
        )

    # For backwards compatibility with version<0.6.0 when multi mount wasn't an option (there was no mounts)
    if not volume_mounts:
        if mount_path:
            if remote:
                # If both remote and mount_path given, no default behavior is expected, we can't assume anything
                # therefore we don't add the v3io volume mount and default to legacy behavior
                return mount_v3io_legacy(
                    name, remote, mount_path, access_key, user, secret
                )
            else:
                # If mount path but no remote, it means the user "counted" on the default remote
                # Back then remote default was ~/ which is /users/<username>, but since we now use multi mount, we're
                # using subpath instead
                user = _resolve_mount_user(user)
                if not user:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "user name/env must be specified when using empty remote and mount_path"
                    )
                volume_mounts = [
                    VolumeMount(path="/v3io", sub_path=""),
                    VolumeMount(path=mount_path, sub_path="users/" + user),
                ]
        else:
            if remote:
                # If remote but no mount path, it means the user "counted" on the default mount path
                # Back then mount_path default was /User, but since the remote was given we can't assume anything
                # therefore we don't add the v3io volume mount and default to legacy behavior
                return mount_v3io_legacy(
                    name, remote, access_key=access_key, user=user, secret=secret
                )
            # not remote and not mounts (and not mount_path) handled by the extended handler

    return mount_v3io_extended(
        name=name,
        remote=remote,
        mounts=volume_mounts,
        access_key=access_key,
        user=user,
        secret=secret,
    )


def mount_v3io_legacy(
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
        mounts=[VolumeMount(path=mount_path, sub_path="")],
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

    Usage::

        train = train_op(...)
        train.apply(use_v3io_cred())
    """

    def _use_v3io_cred(task):
        from os import environ

        from kubernetes import client as k8s_client

        web_api = api or environ.get("V3IO_API") or mlconf.v3io_api
        _user = user or environ.get("V3IO_USERNAME")
        _access_key = access_key or environ.get("V3IO_ACCESS_KEY")
        v3io_framesd = mlconf.v3io_framesd or environ.get("V3IO_FRAMESD")

        return (
            task.add_env_variable(k8s_client.V1EnvVar(name="V3IO_API", value=web_api))
            .add_env_variable(k8s_client.V1EnvVar(name="V3IO_USERNAME", value=_user))
            .add_env_variable(
                k8s_client.V1EnvVar(name="V3IO_ACCESS_KEY", value=_access_key)
            )
            .add_env_variable(
                k8s_client.V1EnvVar(name="V3IO_FRAMESD", value=v3io_framesd)
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
        user = user or environ.get("V3IO_USERNAME")
        if not user:
            raise mlrun.errors.MLRunInvalidArgumentError(
                'user name/env must be specified when using "~" in path'
            )
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
    def __init__(
        self,
        stream_path,
        shards=None,
        retention_in_hours=None,
        create=True,
        endpoint=None,
        access_key=None,
    ):
        v3io_client_kwargs = {}
        if endpoint:
            v3io_client_kwargs["endpoint"] = endpoint
        if access_key:
            v3io_client_kwargs["access_key"] = access_key

        self._v3io_client = v3io.dataplane.Client(**v3io_client_kwargs)
        self._container, self._stream_path = split_path(stream_path)

        if create:

            # this import creates an import loop via the utils module, so putting it in execution path
            from mlrun.utils.helpers import logger

            logger.debug(
                "Creating output stream",
                endpoint=endpoint,
                container=self._container,
                stream_path=self._stream_path,
                shards=shards,
                retention_in_hours=retention_in_hours,
            )

            response = self._v3io_client.create_stream(
                container=self._container,
                path=self._stream_path,
                shard_count=shards or 1,
                retention_period_hours=retention_in_hours or 24,
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
            )
            if not (
                response.status_code == 400 and "ResourceInUse" in str(response.body)
            ):
                response.raise_for_status([409, 204])

    def push(self, data):
        if not isinstance(data, list):
            data = [data]
        records = [{"data": json.dumps(rec)} for rec in data]
        self._v3io_client.put_records(
            container=self._container, path=self._stream_path, records=records
        )


class V3ioStreamClient:
    def __init__(self, url: str, shard_id: int = 0, seek_to: str = None, **kwargs):
        endpoint, stream_path = parse_v3io_path(url)
        seek_options = ["EARLIEST", "LATEST", "TIME", "SEQUENCE"]
        seek_to = seek_to or "LATEST"
        seek_to = seek_to.upper()
        if seek_to not in seek_options:
            raise ValueError(f'seek_to must be one of {", ".join(seek_options)}')

        self._url = url
        self._container, self._stream_path = split_path(stream_path)
        self._shard_id = shard_id
        self._seek_to = seek_to
        self._client = v3io.dataplane.Client(endpoint=endpoint)
        self._seek_done = False
        self._location = ""
        self._kwargs = kwargs

    @property
    def url(self):
        return self._url

    @property
    def shard_id(self):
        return self._shard_id

    def seek(self):
        response = self._client.stream.seek(
            self._container,
            self._stream_path,
            self._shard_id,
            self._seek_to,
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
            **self._kwargs,
        )
        if response.status_code == 404 and "ResourceNotFound" in str(response.body):
            return 0
        response.raise_for_status()
        self._location = response.output.location
        self._seek_done = True
        return response.status_code

    def get_records(self):
        if not self._seek_done:
            resp = self.seek()
            if resp == 0:
                return []
        response = self._client.stream.get_records(
            self._container, self._stream_path, self._shard_id, self._location
        )
        response.raise_for_status()
        self._location = response.output.next_location
        return response.output.records


def create_control_session(url, username, password):
    # for systems without production cert - silence no cert verification WARN
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    if not username or not password:
        raise ValueError("cannot create session key, missing username or password")

    session = requests.Session()
    session.auth = (username, password)
    try:
        auth = session.post(f"{url}/api/sessions", verify=False)
    except OSError as exc:
        raise OSError(f"error: cannot connect to {url}: {exc}")

    if not auth.ok:
        raise OSError(f"failed to create session: {url}, {auth.text}")

    return auth.json()["data"]["id"]


def is_iguazio_endpoint(endpoint_url: str) -> bool:
    # TODO: find a better heuristic
    return ".default-tenant." in endpoint_url


def is_iguazio_session(value: str) -> bool:
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

    if is_iguazio_session(password):
        return username, password, token

    username = username or os.environ.get("V3IO_USERNAME")
    password = password or os.environ.get("V3IO_PASSWORD")
    # V3IO_ACCESS_KEY` is used by other packages like v3io, MLRun also uses it as the access key used to
    # communicate with the API from the client. `MLRUN_AUTH_SESSION` is for when we want
    # different access keys for the 2 usages
    token = (
        token
        or os.environ.get("MLRUN_AUTH_SESSION")
        or os.environ.get("V3IO_ACCESS_KEY")
    )

    # When it's not iguazio endpoint it's one of two options:
    # Enterprise, but we're in the cluster (and not from remote), e.g. url will be something like http://mlrun-api:8080
    # In which we enforce to have access key which is needed for the API auth
    # Open source in which auth is not enabled so no creds needed
    # We don't really have an easy/nice way to differentiate between the two so we're just sending creds anyways
    # (ideally if we could identify we're in enterprise we would have verify here that token and username have value)
    if not is_iguazio_endpoint(api_url):
        return "", "", token
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


def parse_v3io_path(url, suffix="/"):
    """return v3io table path from url"""
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme.lower()
    if scheme != "v3io" and scheme != "v3ios":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "url must start with v3io://[host]/{container}/{path}, got " + url
        )
    endpoint = parsed_url.hostname
    if endpoint:
        if parsed_url.port:
            endpoint += f":{parsed_url.port}"
        prefix = "https" if scheme == "v3ios" else "http"
        endpoint = f"{prefix}://{endpoint}"
    else:
        endpoint = None
    return endpoint, parsed_url.path.strip("/") + suffix
