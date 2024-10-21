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
import datetime
import functools
import re
import time
from typing import Callable, Optional, Union

import semver
from humanfriendly import InvalidTimespan, parse_timespan
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
            has_py_package = (
                not version.prerelease or version.prerelease.find("-") == -1
            )

    return tag, has_py_package


def is_request_from_leader(
    projects_role: Optional[mlrun.common.schemas.ProjectsRole], leader_name: str = None
):
    leader_name = leader_name or mlrun.mlconf.httpdb.projects.leader
    if projects_role and projects_role.value == leader_name:
        return True
    return False


def string_to_timedelta(
    date_str: str, offset: int = 0, raise_on_error: bool = True
) -> Optional[datetime.timedelta]:
    date_str = date_str.strip().lower()
    try:
        seconds = parse_timespan(date_str) + offset
    except InvalidTimespan as exc:
        if raise_on_error:
            raise exc
        return None

    return datetime.timedelta(seconds=seconds)


def lru_cache_with_ttl(maxsize=128, typed=False, ttl_seconds=60):
    """
    Thread-safety least-recently used cache with time-to-live (ttl_seconds) limit.
    https://stackoverflow.com/a/71634221/5257501
    """

    class Result:
        __slots__ = ("value", "death")

        def __init__(self, value, death):
            self.value = value
            self.death = death

    def decorator(func):
        @functools.lru_cache(maxsize=maxsize, typed=typed)
        def cached_func(*args, **kwargs):
            value = func(*args, **kwargs)
            death = time.monotonic() + ttl_seconds
            return Result(value, death)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = cached_func(*args, **kwargs)
            if result.death < time.monotonic():
                result.value = func(*args, **kwargs)
                result.death = time.monotonic() + ttl_seconds
            return result.value

        wrapper.cache_clear = cached_func.cache_clear
        return wrapper

    return decorator


def set_scheduled_object_labels(
    scheduled_object: Union[Optional[dict], Callable], labels: Optional[dict]
) -> None:
    if not isinstance(scheduled_object, dict):
        return
    scheduled_object.setdefault("task", {}).setdefault("metadata", {})["labels"] = (
        labels
    )


def merge_schedule_and_db_schedule_labels(
    labels: Optional[dict],
    scheduled_object: Union[Optional[dict], Callable],
    db_schedule: Optional[mlrun.common.schemas.ScheduleRecord],
) -> tuple[Optional[dict], Union[Optional[dict], Callable]]:
    """
    Merges the provided schedule labels and scheduled object labels with the labels
    from the database schedule. The method ensures that the scheduled object's labels
    are properly aligned with the schedule labels.

    :param labels: The labels of the schedule
    :param scheduled_object: The scheduled object
    :param db_schedule: A ScheduleRecord object from the database, containing the existing labels
                        and scheduled object to be merged
    :return: The merged labels and the updated scheduled object, ensuring alignment between
             provided and database labels
    """
    db_labels = {}
    if db_schedule:
        # convert list[LabelRecord] to dict
        db_schedule_labels = {label.name: label.value for label in db_schedule.labels}
        # merge schedule's labels and scheduled object's labels for object from db
        db_labels = merge_schedule_and_schedule_object_labels(
            db_schedule_labels, db_schedule.scheduled_object
        )

    # merge schedule's labels and scheduled object's labels for passed values
    labels = merge_schedule_and_schedule_object_labels(labels, scheduled_object)

    # if labels are None, then we don't want to overwrite them and labels should remain the same as in db
    # if labels are {} then we do want to overwrite them
    if labels is None and db_schedule:
        labels = db_labels
        # ensure that labels value in db are aligned (for cases when we upgrade from version, where they weren't)
        scheduled_object = db_schedule.scheduled_object
        set_scheduled_object_labels(scheduled_object, db_labels)

    # If schedule object isn't passed,
    # Ensure that schedule_object has the same value as schedule.labels
    if scheduled_object is None and db_schedule:
        scheduled_object = db_schedule.scheduled_object
        set_scheduled_object_labels(scheduled_object, labels)

    return labels, scheduled_object


def merge_schedule_and_schedule_object_labels(
    labels: Optional[dict],
    scheduled_object: Union[Optional[dict], Callable],
) -> Optional[dict]:
    """
    Merges the labels of the scheduled object, giving precedence to the scheduled object labels
    :param labels: The labels of a schedule
    :param scheduled_object: A scheduled object

    :return: Merged labels
    """
    # Ensure scheduled_object is a dictionary-like object
    if not isinstance(scheduled_object, dict):
        return labels

    # Extract the scheduled object labels
    scheduled_object_labels = (
        scheduled_object.get("task", {}).get("metadata", {}).get("labels", {})
    )

    # If labels are empty, no need to update scheduled_object_labels,
    if not labels:
        return scheduled_object_labels

    scheduled_object_labels = scheduled_object_labels or {}

    # Merge labels, giving precedence to scheduled_object_labels
    updated_labels = mlrun.utils.merge_dicts_with_precedence(
        labels, scheduled_object_labels
    )

    # Update the original scheduled_object with the merged labels
    set_scheduled_object_labels(scheduled_object, updated_labels)

    return updated_labels
