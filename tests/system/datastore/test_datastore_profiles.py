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

import pytest

import mlrun
from mlrun.datastore.datastore_profile import DatastoreProfileBasic
from tests.system.base import TestMLRunSystem


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestDatastoreProfile(TestMLRunSystem):
    project_name = "dsprofiles-system-test-project"

    def custom_setup(self):
        pass

    def test_datastore_profile_get(self):
        project = mlrun.get_or_create_project(
            self.project_name, allow_cross_project=True
        )

        profile1 = DatastoreProfileBasic(name="dsname1", public="http://1.1.1.1:1234")

        project.register_datastore_profile(profile1)
        profile = project.get_datastore_profile("dsname1")
        assert profile == profile1

    def test_datastore_profile_list(self):
        project = mlrun.get_or_create_project(
            self.project_name, allow_cross_project=True
        )

        profile1 = DatastoreProfileBasic(name="dsname1", public="http://1.1.1.1:1234")
        profile2 = DatastoreProfileBasic(name="dsname2", public="http://2.2.2.2:1234")

        project.register_datastore_profile(profile1)
        project.register_datastore_profile(profile2)
        profiles = project.list_datastore_profiles()
        assert profiles == [profile1, profile2]

    def test_datastore_profile_delete(self):
        project = mlrun.get_or_create_project(
            self.project_name, allow_cross_project=True
        )

        profile1 = DatastoreProfileBasic(name="dsname1", public="http://1.1.1.1:1234")
        profile2 = DatastoreProfileBasic(name="dsname2", public="http://2.2.2.2:1234")

        project.register_datastore_profile(profile1)
        project.register_datastore_profile(profile2)

        project.delete_datastore_profile("dsname1")
        profiles = project.list_datastore_profiles()
        assert profiles == [profile2]
