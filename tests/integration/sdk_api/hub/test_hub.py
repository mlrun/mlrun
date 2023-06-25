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
import deepdiff

import mlrun
import tests.integration.sdk_api.base


class TestHub(tests.integration.sdk_api.base.TestMLRunIntegration):
    @staticmethod
    def _assert_source_lists_match(expected_response):
        response = mlrun.get_run_db().list_hub_sources()

        exclude_paths = [
            "root['source']['metadata']['updated']",
            "root['source']['metadata']['created']",
        ]
        for i in range(len(expected_response)):
            assert (
                deepdiff.DeepDiff(
                    expected_response[i].dict(),
                    response[i].dict(),
                    exclude_paths=exclude_paths,
                )
                == {}
            )

    def test_hub(self):
        db = mlrun.get_run_db()

        default_source = mlrun.common.schemas.IndexedHubSource(
            index=-1,
            source=mlrun.common.schemas.HubSource.generate_default_source(),
        )
        self._assert_source_lists_match([default_source])

        new_source = mlrun.common.schemas.IndexedHubSource(
            source=mlrun.common.schemas.HubSource(
                metadata=mlrun.common.schemas.HubObjectMetadata(
                    name="source-1", description="a private source"
                ),
                spec=mlrun.common.schemas.HubSourceSpec(
                    path="/local/path/to/source", channel="development"
                ),
            )
        )
        db.create_hub_source(new_source)
        new_source.index = 1
        self._assert_source_lists_match([new_source, default_source])

        new_source_2 = mlrun.common.schemas.IndexedHubSource(
            index=1,
            source=mlrun.common.schemas.HubSource(
                metadata=mlrun.common.schemas.HubObjectMetadata(
                    name="source-2", description="2nd private source"
                ),
                spec=mlrun.common.schemas.HubSourceSpec(
                    path="/local/path/to/source", channel="prod"
                ),
            ),
        )

        db.create_hub_source(new_source_2)
        new_source.index = 2
        self._assert_source_lists_match([new_source_2, new_source, default_source])

        new_source.index = 1
        db.store_hub_source(new_source.source.metadata.name, new_source)
        new_source_2.index = 2
        self._assert_source_lists_match([new_source, new_source_2, default_source])

        db.delete_hub_source("source-1")
        new_source_2.index = 1
        self._assert_source_lists_match([new_source_2, default_source])
