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
import asyncio
from http import HTTPStatus

import httpx
import pytest

import tests.integration.sdk_api.base


class TestFeatureStore(tests.integration.sdk_api.base.TestMLRunIntegration):
    @staticmethod
    def _generate_feature_set(name):
        return {
            "kind": "FeatureSet",
            "metadata": {
                "name": name,
                "labels": {"owner": "saarc", "group": "dev"},
                "tag": "latest",
                "extra_metadata": 100,
            },
            "spec": {
                "entities": [
                    {
                        "name": "ticker",
                        "value_type": "str",
                        "labels": {"label1": "value1"},
                        "extra_entity_field": "here",
                    }
                ],
                "features": [
                    {
                        "name": "time",
                        "value_type": "datetime",
                        "labels": {"label2": "value2"},
                        "extra_feature_field": "there",
                    },
                    {
                        "name": "bid",
                        "value_type": "float",
                        "labels": {"label3": "value3"},
                    },
                    {
                        "name": "ask",
                        "value_type": "time",
                        "labels": {"label4": "value4"},
                    },
                ],
                "extra_spec": True,
            },
            "status": {
                "state": "created",
                "stats": {
                    "time": {
                        "count": "8",
                        "unique": "7",
                        "top": "2016-05-25 13:30:00.222222",
                    }
                },
                "extra_status": {"field1": "value1", "field2": "value2"},
            },
        }

    @pytest.mark.asyncio
    async def test_feature_set_multiple_concurrent_store_operations(self):
        # This test reproduces the issue seen at ML-2004, where multiple concurrent PUT requests for a feature-set
        # created a situation where features & entities were duplicated in the DB (but only when working with mysql).
        async_client = httpx.AsyncClient(base_url=self.base_url)

        project_name = "some-project"
        project = {
            "metadata": {
                "name": project_name,
            }
        }
        response = await async_client.post(
            "projects",
            json=project,
        )
        assert response.status_code == HTTPStatus.CREATED.value

        feature_set_name = "feature_set_1"
        reference = "latest"

        feature_set = self._generate_feature_set(feature_set_name)
        response = await async_client.put(
            f"projects/{project_name}/feature-sets/{feature_set_name}/references/{reference}?versioned=false",
            json=feature_set,
        )
        assert response.status_code == HTTPStatus.OK.value

        # Change the feature-set a bit, so that it will override.
        feature_set["metadata"]["new_metadata"] = True
        request1_task = asyncio.create_task(
            async_client.put(
                f"projects/{project_name}/feature-sets/{feature_set_name}/references/{reference}?versioned=false",
                json=feature_set,
            )
        )
        request2_task = asyncio.create_task(
            async_client.put(
                f"projects/{project_name}/feature-sets/{feature_set_name}/references/{reference}?versioned=false",
                json=feature_set,
            )
        )
        response1, response2 = await asyncio.gather(
            request1_task,
            request2_task,
        )

        assert response1.status_code == HTTPStatus.OK.value
        assert response2.status_code == HTTPStatus.OK.value

        response = await async_client.get(f"projects/{project_name}/features?name=bid")
        assert response.status_code == HTTPStatus.OK.value
        results = response.json()
        assert len(results["features"]) == 1

        response = await async_client.get(
            f"projects/{project_name}/entities?name=ticker"
        )
        assert response.status_code == HTTPStatus.OK.value
        results = response.json()
        assert len(results["entities"]) == 1
