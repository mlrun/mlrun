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

import framework.db.sqldb.models
import services.api.initial_data
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestHub(TestDatabaseBase):
    def test_data_migration_rename_marketplace_kind_to_hub(
        self,
    ):
        # create hub sources
        for i in range(3):
            source_name = f"source-{i}"
            source_dict = {
                "metadata": {
                    "name": source_name,
                },
                "spec": {
                    "path": "/local/path/to/source",
                },
                "kind": "MarketplaceSource",
            }
            # id and index are multiplied by 2 to avoid sqlalchemy unique constraint error
            source = framework.db.sqldb.models.HubSource(
                id=i * 2,
                name=source_name,
                index=i * 2,
            )
            source.full_object = source_dict
            self._db_session.add(source)
            self._db_session.commit()

        # run migration
        services.api.initial_data._rename_marketplace_kind_to_hub(
            self._db, self._db_session
        )

        # check that all hub sources are now of kind 'HubSource'
        hubs = self._db._list_hub_sources_without_transform(self._db_session)
        for hub in hubs:
            hub_dict = hub.full_object
            assert "kind" in hub_dict
            assert hub_dict["kind"] == "HubSource"
