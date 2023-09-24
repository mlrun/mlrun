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
import json
import math
import typing

import sqlalchemy.orm

import mlrun.api
import mlrun.common
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.schedule
import mlrun.errors

Seconds = typing.NewType("Seconds", int)
Minutes = typing.NewType("Minutes", int)

_SECONDS_IN_MINUTE: Seconds = Seconds(60)
_MINUTES_IN_HOUR: Minutes = Minutes(60)


def seconds2minutes(seconds: Seconds) -> Minutes:
    return Minutes(math.ceil(seconds / _SECONDS_IN_MINUTE))


def get_batching_interval_param(intervals_list: typing.List):
    """Convert each value in the intervals list into a float number. None
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


def _add_minutes_offset(
    minute: typing.Optional[typing.Union[int, str]],
    offset: Minutes,
) -> typing.Optional[typing.Union[int, str]]:
    """
    :param minute: the minute specification in the cron schedule, e.g. "0".
    :param offset: the offset in minutes to add to the cron minute specification.
    :return: the minute cron with the offset applied (if supported).
    """
    if minute and (
        (isinstance(minute, str) and str.isdigit(minute)) or isinstance(minute, int)
    ):
        minute = (int(minute) + offset) % _MINUTES_IN_HOUR
    return minute


def convert_to_cron_string(
    cron_trigger: mlrun.common.schemas.schedule.ScheduleCronTrigger,
    minute_delay: Minutes = Minutes(0),
) -> str:
    """Convert the batch interval `ScheduleCronTrigger` into a cron trigger expression"""
    return "{} {} {} * *".format(
        _add_minutes_offset(cron_trigger.minute, minute_delay),
        cron_trigger.hour,
        cron_trigger.day,
    ).replace("None", "*")


def clean_feature_name(feature_name):
    return feature_name.replace(" ", "_").replace("(", "").replace(")", "")


def json_loads_if_not_none(field: typing.Any) -> typing.Any:
    return (
        json.loads(field) if field and field != "null" and field is not None else None
    )


def get_access_key(auth_info: mlrun.common.schemas.AuthInfo):
    """
    Get access key from the current data session. This method is usually used to verify that the session
    is valid and contains an access key.

    param auth_info: The auth info of the request.

    :return: Access key as a string.
    """
    access_key = auth_info.data_session
    if not access_key:
        raise mlrun.errors.MLRunBadRequestError("Data session is missing")
    return access_key


def get_monitoring_parquet_path(
    db_session: sqlalchemy.orm.Session,
    project: str,
    kind: str = mlrun.common.schemas.model_monitoring.FileTargetKind.PARQUET,
) -> str:
    """Get model monitoring parquet target for the current project. The parquet target path is based on the
    project artifact path. If project artifact path is not defined, the parquet target path will be based on MLRun
    artifact path.

    :param db_session: A session that manages the current dialog with the database. Will be used in this function
                       to get the project record from DB.
    :param project:    Project name.
    :param kind:       indicate the kind of the parquet path, can be either stream_parquet or stream_controller_parquet

    :return:           Monitoring parquet target path.
    """

    # Get the artifact path from the project record that was stored in the DB
    project_obj = mlrun.api.crud.projects.Projects().get_project(
        session=db_session, name=project
    )
    artifact_path = project_obj.spec.artifact_path
    # Generate monitoring parquet path value
    parquet_path = mlrun.mlconf.get_model_monitoring_file_target_path(
        project=project,
        kind=kind,
        target="offline",
        artifact_path=artifact_path,
    )
    return parquet_path


def get_stream_path(project: str = None, application_name: str = None):
    """
    Get stream path from the project secret. If wasn't set, take it from the system configurations

    :param project:             Project name.
    :param application_name:    Application name, None for model_monitoring_stream.

    :return:                    Monitoring stream path to the relevant application.
    """

    stream_uri = mlrun.api.crud.secrets.Secrets().get_project_secret(
        project=project,
        provider=mlrun.common.schemas.secret.SecretProviderName.kubernetes,
        allow_secrets_from_k8s=True,
        secret_key=mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
        if application_name is None
        else "",
    ) or mlrun.mlconf.get_model_monitoring_file_target_path(
        project=project,
        kind=mlrun.common.schemas.model_monitoring.FileTargetKind.STREAM,
        target="online",
        application_name=application_name,
    )

    return mlrun.common.model_monitoring.helpers.parse_monitoring_stream_path(
        stream_uri=stream_uri, project=project, application_name=application_name
    )
