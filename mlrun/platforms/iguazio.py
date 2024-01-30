# Copyright 2023 Iguazio
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
import urllib
from collections import namedtuple
from datetime import datetime
from http import HTTPStatus
from urllib.parse import urlparse

import requests
import urllib3
import v3io

import mlrun.errors
from mlrun.errors import err_to_str
from mlrun.utils import dict_to_json

_cached_control_session = None

VolumeMount = namedtuple("Mount", ["path", "sub_path"])


def _enrich_and_validate_v3io_mounts(remote="", volume_mounts=None, user=""):
    if remote and not volume_mounts:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "volume_mounts must be specified when remote is given"
        )

    # Empty remote & volume_mounts defaults are volume mounts of /v3io and /User
    if not remote and not volume_mounts:
        user = _resolve_mount_user(user)
        if not user:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "user name/env must be specified when using empty remote and volume_mounts"
            )
        volume_mounts = [
            VolumeMount(path="/v3io", sub_path=""),
            VolumeMount(path="/User", sub_path="users/" + user),
        ]

    if not isinstance(volume_mounts, list) and any(
        [not isinstance(x, VolumeMount) for x in volume_mounts]
    ):
        raise TypeError("mounts should be a list of Mount")

    return volume_mounts, user


def _resolve_mount_user(user=None):
    return user or os.environ.get("V3IO_USERNAME")


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
    user = user or environ.get("V3IO_USERNAME")
    if user:
        opts["dirsToCreate"] = f'[{{"name": "users//{user}", "permissions": 488}}]'

    remote = str(remote)

    if remote.startswith("~/"):
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
        mock=False,
        **kwargs,  # to avoid failing on extra parameters
    ):
        v3io_client_kwargs = {}
        if endpoint:
            v3io_client_kwargs["endpoint"] = endpoint
        if access_key:
            v3io_client_kwargs["access_key"] = access_key

        self._v3io_client = v3io.dataplane.Client(**v3io_client_kwargs)
        self._container, self._stream_path = split_path(stream_path)
        self._mock = mock
        self._mock_queue = []

        if create and not mock:
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
            response = self._v3io_client.stream.create(
                container=self._container,
                stream_path=self._stream_path,
                shard_count=shards or 1,
                retention_period_hours=retention_in_hours or 24,
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
            )
            if not (
                response.status_code == 400 and "ResourceInUse" in str(response.body)
            ):
                response.raise_for_status([409, 204])

    def push(self, data):
        def dump_record(rec):
            if not isinstance(rec, (str, bytes)):
                return dict_to_json(rec)
            return str(rec)

        if not isinstance(data, list):
            data = [data]
        records = [{"data": dump_record(rec)} for rec in data]
        if self._mock:
            # for mock testing
            self._mock_queue.extend(records)
        else:
            self._v3io_client.stream.put_records(
                container=self._container,
                stream_path=self._stream_path,
                records=records,
            )


class HTTPOutputStream:
    """HTTP output source that usually used for CE mode and debugging process"""

    def __init__(self, stream_path: str):
        self._stream_path = stream_path

    def push(self, data):
        def dump_record(rec):
            if isinstance(rec, bytes):
                return rec

            if not isinstance(rec, str):
                rec = dict_to_json(rec)

            return rec.encode("UTF-8")

        if not isinstance(data, list):
            data = [data]

        for record in data:
            # Convert the new record to the required format
            serialized_record = dump_record(record)
            response = requests.post(self._stream_path, data=serialized_record)
            if not response:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"API call failed push a new record through {self._stream_path}, "
                    f"status {response.status_code}: {response.reason}"
                )


class KafkaOutputStream:
    def __init__(
        self,
        topic,
        brokers,
        producer_options=None,
        mock=False,
    ):
        self._kafka_producer = None
        self._topic = topic
        self._brokers = brokers
        self._producer_options = producer_options or {}

        self._mock = mock
        self._mock_queue = []

        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return

        import kafka

        self._kafka_producer = kafka.KafkaProducer(
            bootstrap_servers=self._brokers,
            **self._producer_options,
        )

        self._initialized = True

    def push(self, data):
        self._lazy_init()

        def dump_record(rec):
            if isinstance(rec, bytes):
                return rec

            if not isinstance(rec, str):
                rec = dict_to_json(rec)

            return rec.encode("UTF-8")

        if not isinstance(data, list):
            data = [data]

        if self._mock:
            # for mock testing
            self._mock_queue.extend(data)
        else:
            for record in data:
                serialized_record = dump_record(record)
                self._kafka_producer.send(self._topic, serialized_record)


class V3ioStreamClient:
    def __init__(self, url: str, shard_id: int = 0, seek_to: str = None, **kwargs):
        endpoint, stream_path = parse_path(url)
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
        raise OSError(f"error: cannot connect to {url}: {err_to_str(exc)}")

    if not auth.ok:
        raise OSError(f"failed to create session: {url}, {auth.text}")

    return auth.json()["data"]["id"]


def is_iguazio_endpoint(endpoint_url: str) -> bool:
    # TODO: find a better heuristic
    return ".default-tenant." in endpoint_url


def is_iguazio_session(value: str) -> bool:
    # TODO: find a better heuristic
    return len(value) > 20 and "-" in value


def is_iguazio_session_cookie(session_cookie: str) -> bool:
    if not session_cookie.strip():
        return False

    # decode url encoded cookie
    # from: j%3A%7B%22sid%22%3A%20%22946b0749-5c40-4837-a4ac-341d295bfaf7%22%7D
    # to:   j:{"sid":"946b0749-5c40-4837-a4ac-341d295bfaf7"}
    try:
        unqouted_cookie = urllib.parse.unquote(session_cookie.strip())
        if not unqouted_cookie.startswith("j:"):
            return is_iguazio_session(session_cookie)
        return json.loads(unqouted_cookie[2:])["sid"] is not None
    except Exception:
        return False


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
        # can't use mlrun.runtimes.constants.FunctionEnvironmentVariables.auth_session cause this is running in the
        # import execution path (when we're initializing the run db) and therefore we can't import mlrun.runtimes
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


def parse_path(url, suffix="/"):
    """return endpoint and table path from url"""
    parsed_url = urlparse(url)
    if parsed_url.netloc:
        scheme = parsed_url.scheme.lower()
        if scheme == "v3ios":
            prefix = "https"
        elif scheme == "v3io":
            prefix = "http"
        elif scheme == "redis":
            prefix = "redis"
        elif scheme == "rediss":
            prefix = "rediss"
        elif scheme == "ds":
            prefix = "ds"
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "url must start with v3io/v3ios/redis/rediss, got " + url
            )
        endpoint = f"{prefix}://{parsed_url.netloc}"
    else:
        # no netloc is mainly when using v3io (v3io:///) and expecting the url to be resolved automatically from env or
        # config
        endpoint = None
    return endpoint, parsed_url.path.strip("/") + suffix


def sanitize_username(username: str):
    """
    The only character an Iguazio username may have that is not valid for k8s usage is underscore (_)
    So simply replace it with dash
    """
    return username.replace("_", "-")
