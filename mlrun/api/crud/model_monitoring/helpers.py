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
#
import json
import typing

import mlrun.common.schemas.schedule


def get_batching_interval_param(intervals_list: typing.List):
    """Converting each value in the intervals list into a float number. None
    Values will be converted into 0.0.

    param intervals_list: A list of values based on the ScheduleCronTrigger expression. Note that at the moment
                          it supports minutes, hours, and days. e.g. [0, '*/1', None] represents on the hour
                          every hour.

    :return: A tuple of:
             [0] = minutes interval as a float
             [1] = hours interval as a float
             [2] = days interval as a float
    """
    return tuple(
        [
            0.0
            if isinstance(interval, (float, int)) or interval is None
            else float(f"0{interval.partition('/')[-1]}")
            for interval in intervals_list
        ]
    )


def convert_to_cron_string(
    cron_trigger: mlrun.common.schemas.schedule.ScheduleCronTrigger,
):
    """Converting the batch interval `ScheduleCronTrigger` into a cron trigger expression"""
    return "{} {} {} * *".format(
        cron_trigger.minute, cron_trigger.hour, cron_trigger.day
    ).replace("None", "*")


def clean_feature_name(feature_name):
    return feature_name.replace(" ", "_").replace("(", "").replace(")", "")


def json_loads_if_not_none(field: typing.Any) -> typing.Any:
    return (
        json.loads(field) if field and field != "null" and field is not None else None
    )
