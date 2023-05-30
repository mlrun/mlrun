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
import warnings
from typing import Union

import mlrun
import mlrun.model
import mlrun.model_monitoring.constants as model_monitoring_constants
import mlrun.platforms.iguazio
from mlrun.api.schemas.schedule import ScheduleCronTrigger
from mlrun.config import is_running_as_api


def parse_model_endpoint_project_prefix(path: str, project_name: str):
    return path.split(project_name, 1)[0] + project_name


def parse_model_endpoint_store_prefix(store_prefix: str):
    endpoint, parsed_url = mlrun.platforms.iguazio.parse_path(store_prefix)
    container, path = parsed_url.split("/", 1)
    return endpoint, container, path


def set_project_model_monitoring_credentials(access_key: str, project: str = None):
    """Set the credentials that will be used by the project's model monitoring
    infrastructure functions.
    The supplied credentials must have data access
    :param access_key: Model Monitoring access key for managing user permissions.
    :param project: The name of the model monitoring project.
    """
    mlrun.get_run_db().create_project_secrets(
        project=project or mlrun.mlconf.default_project,
        provider=mlrun.api.schemas.SecretProviderName.kubernetes,
        secrets={model_monitoring_constants.ProjectSecretKeys.ACCESS_KEY: access_key},
    )


class TrackingPolicy(mlrun.model.ModelObj):
    """
    Modified model monitoring configurations. By using TrackingPolicy, the user can apply his model monitoring
    requirements, such as setting the scheduling policy of the model monitoring batch job or changing the image of the
    model monitoring stream.
    """

    _dict_fields = [
        "default_batch_image",
        "stream_image",
    ]

    def __init__(
        self,
        default_batch_intervals: Union[ScheduleCronTrigger, str] = ScheduleCronTrigger(
            minute="0", hour="*/1"
        ),
        default_batch_image: str = "mlrun/mlrun",
        stream_image: str = "mlrun/mlrun",
    ):
        """
        Initialize TrackingPolicy object.
        :param default_batch_intervals:     Model monitoring batch scheduling policy. By default, executed on the hour
                                            every hour. Can be either a string or a ScheduleCronTrigger object. The
                                            string time format is based on ScheduleCronTrigger expression:
                                            minute, hour, day of month, month, day of week. It will be converted into
                                            a ScheduleCronTrigger object.
        :param default_batch_image:         The default image of the model monitoring batch job. By default, the image
                                            is mlrun/mlrun.
        :param stream_image:                The image of the model monitoring stream real-time function. By default,
                                            the image is mlrun/mlrun.
        """
        if isinstance(default_batch_intervals, str):
            default_batch_intervals = ScheduleCronTrigger.from_crontab(
                default_batch_intervals
            )
        self.default_batch_intervals = default_batch_intervals
        self.default_batch_image = default_batch_image
        self.stream_image = stream_image

    @classmethod
    def from_dict(cls, struct=None, fields=None, deprecated_fields: dict = None):
        new_obj = super().from_dict(
            struct, fields=cls._dict_fields, deprecated_fields=deprecated_fields
        )
        # Convert default batch interval into ScheduleCronTrigger object
        if model_monitoring_constants.EventFieldType.DEFAULT_BATCH_INTERVALS in struct:
            if isinstance(
                struct[
                    model_monitoring_constants.EventFieldType.DEFAULT_BATCH_INTERVALS
                ],
                str,
            ):
                new_obj.default_batch_intervals = ScheduleCronTrigger.from_crontab(
                    struct[
                        model_monitoring_constants.EventFieldType.DEFAULT_BATCH_INTERVALS
                    ]
                )
            else:
                new_obj.default_batch_intervals = ScheduleCronTrigger.parse_obj(
                    struct[
                        model_monitoring_constants.EventFieldType.DEFAULT_BATCH_INTERVALS
                    ]
                )
        return new_obj

    def to_dict(self, fields=None, exclude=None):
        struct = super().to_dict(
            fields,
            exclude=[model_monitoring_constants.EventFieldType.DEFAULT_BATCH_INTERVALS],
        )
        if self.default_batch_intervals:
            struct[
                model_monitoring_constants.EventFieldType.DEFAULT_BATCH_INTERVALS
            ] = self.default_batch_intervals.dict()
        return struct


def get_connection_string(project: str = None):
    """Get endpoint store connection string from the project secret.
    If wasn't set, take it from the system configurations"""
    if is_running_as_api():
        # Running on API server side
        import mlrun.api.crud.secrets
        import mlrun.api.schemas

        return (
            mlrun.api.crud.secrets.Secrets().get_project_secret(
                project=project,
                provider=mlrun.api.schemas.secret.SecretProviderName.kubernetes,
                allow_secrets_from_k8s=True,
                secret_key=model_monitoring_constants.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION,
            )
            or mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection
        )
    else:
        # Running on stream server side
        import mlrun

        return (
            mlrun.get_secret_or_env(
                model_monitoring_constants.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION
            )
            or mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection
        )


def get_stream_path(project: str = None):
    # TODO: This function (as well as other methods in this file) includes both client and server side code. We will
    #  need to refactor and adjust this file in the future.
    """Get stream path from the project secret. If wasn't set, take it from the system configurations"""

    if is_running_as_api():
        # Running on API server side
        import mlrun.api.crud.secrets
        import mlrun.api.schemas

        stream_uri = mlrun.api.crud.secrets.Secrets().get_project_secret(
            project=project,
            provider=mlrun.api.schemas.secret.SecretProviderName.kubernetes,
            allow_secrets_from_k8s=True,
            secret_key=model_monitoring_constants.ProjectSecretKeys.STREAM_PATH,
        ) or mlrun.mlconf.get_model_monitoring_file_target_path(
            project=project,
            kind=model_monitoring_constants.FileTargetKind.STREAM,
            target="online",
        )

    else:
        import mlrun

        stream_uri = mlrun.get_secret_or_env(
            model_monitoring_constants.ProjectSecretKeys.STREAM_PATH
        ) or mlrun.mlconf.get_model_monitoring_file_target_path(
            project=project,
            kind=model_monitoring_constants.FileTargetKind.STREAM,
            target="online",
        )

    if stream_uri.startswith("kafka://"):
        if "?topic" in stream_uri:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Custom kafka topic is not allowed"
            )
        # Add topic to stream kafka uri
        stream_uri += f"?topic=monitoring_stream_{project}"

    elif stream_uri.startswith("v3io://") and mlrun.mlconf.is_ce_mode():
        # V3IO is not supported in CE mode, generating a default http stream path
        stream_uri = mlrun.mlconf.model_endpoint_monitoring.default_http_sink

    return stream_uri


def validate_old_schema_fields(endpoint: dict):
    """
    Replace default null values for `error_count` and `metrics` for users that logged a model endpoint before 1.3.0.
    In addition, this function also validates that the key name of the endpoint unique id is `uid` and not
     `endpoint_id` that has been used before 1.3.0.

    Leaving here for backwards compatibility which related to the model endpoint schema.

    :param endpoint: An endpoint flattened dictionary.
    """
    warnings.warn(
        "This will be deprecated in 1.3.0, and will be removed in 1.5.0",
        # TODO: In 1.3.0 do changes in examples & demos In 1.5.0 remove
        FutureWarning,
    )

    # Validate default value for `error_count`
    # For backwards compatibility reasons, we validate that the model endpoint includes the `error_count` key
    if (
        model_monitoring_constants.EventFieldType.ERROR_COUNT in endpoint
        and endpoint[model_monitoring_constants.EventFieldType.ERROR_COUNT] == "null"
    ):
        endpoint[model_monitoring_constants.EventFieldType.ERROR_COUNT] = "0"

    # Validate default value for `metrics`
    # For backwards compatibility reasons, we validate that the model endpoint includes the `metrics` key
    if (
        model_monitoring_constants.EventFieldType.METRICS in endpoint
        and endpoint[model_monitoring_constants.EventFieldType.METRICS] == "null"
    ):
        endpoint[model_monitoring_constants.EventFieldType.METRICS] = json.dumps(
            {
                model_monitoring_constants.EventKeyMetrics.GENERIC: {
                    model_monitoring_constants.EventLiveStats.LATENCY_AVG_1H: 0,
                    model_monitoring_constants.EventLiveStats.PREDICTIONS_PER_SECOND: 0,
                }
            }
        )
    # Validate key `uid` instead of `endpoint_id`
    # For backwards compatibility reasons, we replace the `endpoint_id` with `uid` which is the updated key name
    if model_monitoring_constants.EventFieldType.ENDPOINT_ID in endpoint:
        endpoint[model_monitoring_constants.EventFieldType.UID] = endpoint[
            model_monitoring_constants.EventFieldType.ENDPOINT_ID
        ]
