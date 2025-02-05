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

from typing import Any

import pytest
from typing_extensions import TypeAlias

import mlrun
from mlrun import MlrunProject
from mlrun.datastore.datastore_profile import (
    DatastoreProfile,
    DatastoreProfileKafkaSource,
    DatastoreProfileV3io,
    TDEngineDatastoreProfile,
)
from tests.system.base import TestMLRunSystem

_ProfilesMap: TypeAlias = dict[str, type[DatastoreProfile]]

_DS_TYPE_TO_DS_PROFILE: _ProfilesMap = {
    "v3io": DatastoreProfileV3io,
    "taosws": TDEngineDatastoreProfile,
    "kafka_source": DatastoreProfileKafkaSource,
}


@pytest.mark.model_monitoring
class TestMLRunSystemModelMonitoring(TestMLRunSystem):
    project: MlrunProject
    mm_tsdb_profile: DatastoreProfile
    mm_stream_profile: DatastoreProfile

    @staticmethod
    def _get_profile(profile_data: Any, profiles_map: _ProfilesMap) -> DatastoreProfile:
        if isinstance(profile_data, dict):
            ds_type = profile_data.get("type")
            if ds_type in profiles_map:
                return profiles_map[ds_type].parse_obj(profile_data)
            raise ValueError(
                f"Unsupported datastore type: '{ds_type}', expected one of {list(profiles_map)}"
            )
        raise ValueError("The model monitoring profile data is not a dictionary")

    @classmethod
    def get_tsdb_profile(cls, profile_data: dict[str, Any]) -> DatastoreProfile:
        return cls._get_profile(
            profile_data,
            {type_: _DS_TYPE_TO_DS_PROFILE[type_] for type_ in ("v3io", "taosws")},
        )

    @classmethod
    def get_stream_profile(cls, profile_data: dict[str, Any]) -> DatastoreProfile:
        profile = cls._get_profile(
            profile_data,
            {
                type_: _DS_TYPE_TO_DS_PROFILE[type_]
                for type_ in ("v3io", "kafka_source")
            },
        )
        if isinstance(profile, DatastoreProfileV3io):
            # Populate the V3IO access key for the stream profile
            profile.v3io_access_key = (
                profile.v3io_access_key or mlrun.mlconf.get_v3io_access_key()
            )
        return profile

    @classmethod
    def set_mm_profiles(cls):
        cls.mm_tsdb_profile = cls.get_tsdb_profile(cls.mm_tsdb_profile_data)
        cls.mm_stream_profile = cls.get_stream_profile(cls.mm_stream_profile_data)

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.set_mm_profiles()

    def set_mm_credentials(self) -> None:
        self.project.register_datastore_profile(self.mm_tsdb_profile)
        self.project.register_datastore_profile(self.mm_stream_profile)
        self.project.set_model_monitoring_credentials(
            tsdb_profile_name=self.mm_tsdb_profile.name,
            stream_profile_name=self.mm_stream_profile.name,
        )
