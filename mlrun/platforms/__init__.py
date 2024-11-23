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
from pprint import pprint
from time import sleep
from typing import Optional

import mlrun.runtimes.mounts as mlrun_mounts

from .iguazio import (
    V3ioStreamClient,
    add_or_refresh_credentials,
    is_iguazio_session_cookie,
)

# For backwards compatibility
VolumeMount = mlrun_mounts.VolumeMount
auto_mount = mlrun_mounts.auto_mount
mount_configmap = mlrun_mounts.mount_configmap
mount_hostpath = mlrun_mounts.mount_hostpath
mount_pvc = mlrun_mounts.mount_pvc
mount_s3 = mlrun_mounts.mount_s3
mount_secret = mlrun_mounts.mount_secret
mount_v3io = mlrun_mounts.mount_v3io
set_env_variables = mlrun_mounts.set_env_variables
v3io_cred = mlrun_mounts.v3io_cred
# eof 'For backwards compatibility'


def watch_stream(
    url,
    shard_ids: Optional[list] = None,
    seek_to: Optional[str] = None,
    interval=None,
    is_json=False,
    **kwargs,
):
    """watch on a v3io stream and print data every interval

    example::

        watch_stream("v3io:///users/admin/mystream")

    :param url:        stream url
    :param shard_ids:  range or list of shard IDs
    :param seek_to:    where to start/seek ('EARLIEST', 'LATEST', 'TIME', 'SEQUENCE')
    :param interval:   watch interval time in seconds, 0 to run once and return
    :param is_json:    indicate the payload is json (will be deserialized)
    """
    interval = 3 if interval is None else interval
    shard_ids = shard_ids or [0]
    if isinstance(shard_ids, int):
        shard_ids = [shard_ids]
    watchers = [
        V3ioStreamClient(url, shard_id, seek_to, **kwargs)
        for shard_id in list(shard_ids)
    ]
    while True:
        for watcher in watchers:
            records = watcher.get_records()
            for record in records:
                print(
                    f"{watcher.url}:{watcher.shard_id} (#{record.sequence_number}) >> "
                )
                data = json.loads(record.data) if is_json else record.data.decode()
                pprint(data)
        if interval <= 0:
            break
        sleep(interval)
