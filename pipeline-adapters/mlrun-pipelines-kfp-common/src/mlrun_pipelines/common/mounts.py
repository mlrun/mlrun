# Copyright 2024 Iguazio
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
from collections import namedtuple

import mlrun.errors

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
