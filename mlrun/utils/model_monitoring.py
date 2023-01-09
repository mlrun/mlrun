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

from typing import Optional, Union

import mlrun
import mlrun.model
import mlrun.model_monitoring.constants as model_monitoring_constants
import mlrun.platforms.iguazio
from mlrun.api.schemas.schedule import ScheduleCronTrigger


def parse_model_endpoint_project_prefix(path: str, project_name: str):
    return path.split(project_name, 1)[0] + project_name


def parse_model_endpoint_store_prefix(store_prefix: str):
    endpoint, parsed_url = mlrun.platforms.iguazio.parse_path(store_prefix)
    container, path = parsed_url.split("/", 1)
    return endpoint, container, path


def set_project_model_monitoring_credentials(
    access_key: str, project: Optional[str] = None
):
    """Set the credentials that will be used by the project's model monitoring
    infrastructure functions.
    The supplied credentials must have data access

    :param access_key: Model Monitoring access key for managing user permissions.
    :param project: The name of the model monitoring project.
    """
    mlrun.get_run_db().create_project_secrets(
        project=project or mlrun.mlconf.default_project,
        provider=mlrun.api.schemas.SecretProviderName.kubernetes,
        secrets={"MODEL_MONITORING_ACCESS_KEY": access_key},
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
            new_obj.default_batch_intervals = ScheduleCronTrigger.from_crontab(
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
