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
#
import asyncio
import re
from typing import Optional

import semver
from timelength import TimeLength

import mlrun
import mlrun.common.schemas
from mlrun.utils import logger


def ensure_running_on_chief(function):
    """
    The motivation of this function is to catch development bugs in which we are accidentally using functions / flows
    that are supposed to run only in chief instance and by mistake got involved in a worker instance flow.

    Note that there is an option to disable this behavior, its not recommended at all, because it can cause the
    cluster to get out of synchronization.
    """

    def _ensure_running_on_chief():
        if (
            mlrun.mlconf.httpdb.clusterization.role
            != mlrun.common.schemas.ClusterizationRole.chief
        ):
            if (
                mlrun.mlconf.httpdb.clusterization.ensure_function_running_on_chief_mode
                == "enabled"
            ):
                message = (
                    f"{function.__name__} is supposed to run only on chief, re-route."
                )
                raise mlrun.errors.MLRunConflictError(message)
            else:
                logger.warning(
                    f"running {function.__name__} chief function on worker",
                    fail_mode=mlrun.mlconf.httpdb.clusterization.ensure_function_running_on_chief_mode,
                )

    def wrapper(*args, **kwargs):
        _ensure_running_on_chief()
        return function(*args, **kwargs)

    async def async_wrapper(*args, **kwargs):
        _ensure_running_on_chief()
        return await function(*args, **kwargs)

    if asyncio.iscoroutinefunction(function):
        return async_wrapper

    # ensure method name is preserved
    wrapper.__name__ = function.__name__

    return wrapper


def minimize_project_schema(
    project: mlrun.common.schemas.Project,
) -> mlrun.common.schemas.Project:
    project.spec.functions = None
    project.spec.workflows = None
    project.spec.artifacts = None
    return project


def time_string_to_seconds(time_str: str, min_seconds: int = 60) -> Optional[int]:
    if not time_str:
        return None

    if time_str == "-1":
        return -1

    parsed_length = TimeLength(time_str, strict=True)
    total_seconds = parsed_length.to_seconds()
    if total_seconds < min_seconds:
        raise ValueError(f"Invalid time string {time_str}, must be at least 1 minute")

    return total_seconds


def extract_image_tag(image_reference):
    # This matches any word character,dots,hyphens after a colon (:) anchored to the end of the string
    pattern = r"(?<=:)[\w.-]+$"
    match = re.search(pattern, image_reference)

    tag = None
    is_semver = False
    has_py_package = False
    if match:
        tag = match.group()
        is_semver = semver.Version.is_valid(tag)

    if is_semver:
        version = semver.Version.parse(tag)
        # If the version is a prerelease, and it has a hyphen, it means it's a feature branch build
        has_py_package = not version.prerelease or version.prerelease.find("-") == -1

    return tag, has_py_package
