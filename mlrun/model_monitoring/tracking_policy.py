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

import warnings
from typing import Optional, Union

import mlrun.common.schemas.schedule
import mlrun.model


class TrackingPolicy(mlrun.model.ModelObj):
    """
    Modified model monitoring configurations. By using TrackingPolicy, the user can apply his model monitoring
    requirements, such as setting the scheduling policy of the model monitoring batch job or changing the image of the
    model monitoring stream.
    """

    _dict_fields = ["default_batch_image", "stream_image", "application_batch"]

    def __init__(
        self,
        default_batch_intervals: Union[
            mlrun.common.schemas.schedule.ScheduleCronTrigger, str
        ] = mlrun.common.schemas.schedule.ScheduleCronTrigger(minute="0", hour="*/1"),
        default_batch_image: str = "mlrun/mlrun",
        stream_image: str = "mlrun/mlrun",
        base_period: int = 10,
        default_controller_image: str = "mlrun/mlrun",
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
        :param base_period:                 Minutes to determine the frequency in which the model monitoring controller
                                            job is running. By default, the base period is 10 minutes.
        :param default_controller_image:    The default image of the model monitoring controller job. Note that the
                                            writer function, which is a real time nuclio functino, will be deployed
                                            with the same image. By default, the image is mlrun/mlrun.
        """
        warnings.warn(
            "The `TrackingPolicy` class is deprecated from version 1.7.0 and is not "
            "used anymore. It will be removed in 1.10.0.",
            FutureWarning,
        )

        if isinstance(default_batch_intervals, str):
            default_batch_intervals = (
                mlrun.common.schemas.schedule.ScheduleCronTrigger.from_crontab(
                    default_batch_intervals
                )
            )
        self.default_batch_intervals = default_batch_intervals
        self.default_batch_image = default_batch_image
        self.stream_image = stream_image
        self.base_period = base_period
        self.default_controller_image = default_controller_image

    @classmethod
    def from_dict(
        cls, struct=None, fields=None, deprecated_fields: Optional[dict] = None
    ):
        new_obj = super().from_dict(
            struct, fields=cls._dict_fields, deprecated_fields=deprecated_fields
        )
        # Convert default batch interval into ScheduleCronTrigger object
        if (
            mlrun.common.schemas.model_monitoring.EventFieldType.DEFAULT_BATCH_INTERVALS
            in struct
        ):
            if isinstance(
                struct[
                    mlrun.common.schemas.model_monitoring.EventFieldType.DEFAULT_BATCH_INTERVALS
                ],
                str,
            ):
                new_obj.default_batch_intervals = mlrun.common.schemas.schedule.ScheduleCronTrigger.from_crontab(
                    struct[
                        mlrun.common.schemas.model_monitoring.EventFieldType.DEFAULT_BATCH_INTERVALS
                    ]
                )
            else:
                new_obj.default_batch_intervals = mlrun.common.schemas.schedule.ScheduleCronTrigger.parse_obj(
                    struct[
                        mlrun.common.schemas.model_monitoring.EventFieldType.DEFAULT_BATCH_INTERVALS
                    ]
                )
        return new_obj

    def to_dict(
        self,
        fields: Optional[list] = None,
        exclude: Optional[list] = None,
        strip: bool = False,
    ):
        struct = super().to_dict(
            fields,
            exclude=[
                mlrun.common.schemas.model_monitoring.EventFieldType.DEFAULT_BATCH_INTERVALS
            ],
            strip=strip,
        )
        if self.default_batch_intervals:
            struct[
                mlrun.common.schemas.model_monitoring.EventFieldType.DEFAULT_BATCH_INTERVALS
            ] = self.default_batch_intervals.dict()
        return struct
