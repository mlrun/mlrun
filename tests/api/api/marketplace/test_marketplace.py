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
import pathlib
import random
from http import HTTPStatus

import deepdiff
import yaml
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.schemas
import tests.api.conftest
from mlrun.config import config


def _generate_source_dict(index, name, credentials=None):
    path = str(pathlib.Path(__file__).absolute().parent)

    return {
        "index": index,
        "source": {
            "kind": "MarketplaceSource",
            "metadata": {"name": name, "description": "A test", "labels": None},
            "spec": {"path": path, "channel": "catalog", "credentials": credentials},
            "status": {"state": "created"},
        },
    }


def _assert_sources_in_correct_order(client, expected_order, exclude_paths=None):
    exclude_paths = exclude_paths or [
        "root['metadata']['updated']",
        "root['metadata']['created']",
    ]
    response = client.get("marketplace/sources")
    assert response.status_code == HTTPStatus.OK.value
    json_response = response.json()
    # Default source is not in the expected data
    assert len(json_response) == len(expected_order) + 1
    for source in json_response:
        if source["index"] > 0:
            assert (
                deepdiff.DeepDiff(
                    source["source"],
                    expected_order[source["index"]]["source"],
                    exclude_paths=exclude_paths,
                )
                == {}
            )


def test_marketplace_source_apis(
    db: Session,
    client: TestClient,
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    # Make sure the default source is there.
    response = client.get("marketplace/sources")
    assert response.status_code == HTTPStatus.OK.value
    json_response = response.json()
    assert (
        len(json_response) == 1
        and json_response[0]["index"] == -1
        and json_response[0]["source"]["metadata"]["name"]
        == config.marketplace.default_source.name
    )

    source_1 = _generate_source_dict(1, "source_1")
    response = client.post("marketplace/sources", json=source_1)
    assert response.status_code == HTTPStatus.CREATED.value

    # Modify existing source with a new field
    source_1["source"]["metadata"]["something_new"] = 42
    response = client.put("marketplace/sources/source_1", json=source_1)
    assert response.status_code == HTTPStatus.OK.value
    exclude_paths = ["root['metadata']['updated']", "root['metadata']['created']"]
    assert (
        deepdiff.DeepDiff(
            response.json()["source"], source_1["source"], exclude_paths=exclude_paths
        )
        == {}
    )

    # Insert in 1st place, pushing source_1 to be #2
    source_2 = _generate_source_dict(1, "source_2")
    response = client.put("marketplace/sources/source_2", json=source_2)
    assert response.status_code == HTTPStatus.OK.value

    # Insert last, making it #3
    source_3 = _generate_source_dict(-1, "source_3")
    response = client.post("marketplace/sources", json=source_3)
    assert response.status_code == HTTPStatus.CREATED.value

    expected_response = {
        1: source_2,
        2: source_1,
        3: source_3,
    }
    _assert_sources_in_correct_order(client, expected_response)

    # Change order for existing source (3->1)
    source_3["index"] = 1
    response = client.put("marketplace/sources/source_3", json=source_3)
    assert response.status_code == HTTPStatus.OK.value
    expected_response = {
        1: source_3,
        2: source_2,
        3: source_1,
    }
    _assert_sources_in_correct_order(client, expected_response)

    response = client.delete("marketplace/sources/source_2")
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    expected_response = {
        1: source_3,
        2: source_1,
    }
    _assert_sources_in_correct_order(client, expected_response)

    # Negative tests
    # Try to delete the default source.
    response = client.delete(
        f"marketplace/sources/{config.marketplace.default_source.name}"
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST.value
    # Try to store an object with invalid order
    source_2["index"] = 42
    response = client.post("marketplace/sources", json=source_2)
    assert response.status_code == HTTPStatus.BAD_REQUEST.value


def test_marketplace_credentials_removed_from_db(
    db: Session, client: TestClient, k8s_secrets_mock: tests.api.conftest.K8sSecretsMock
) -> None:
    # Validate that a source with credentials is stored (and retrieved back) without them, while the creds
    # are stored in the k8s secret.
    credentials = {"secret1": "value1", "another-secret": "42"}
    source_1 = _generate_source_dict(-1, "source_1", credentials)
    response = client.post("marketplace/sources", json=source_1)
    assert response.status_code == HTTPStatus.CREATED.value

    response = client.get("marketplace/sources/source_1")
    assert response.status_code == HTTPStatus.OK.value
    object_dict = response.json()

    expected_response = source_1["source"]
    expected_response["spec"]["credentials"] = None
    exclude_paths = ["root['metadata']['updated']", "root['metadata']['created']"]
    assert (
        deepdiff.DeepDiff(
            expected_response, object_dict["source"], exclude_paths=exclude_paths
        )
        == {}
    )
    expected_credentials = {
        mlrun.api.crud.Marketplace()._generate_credentials_secret_key(
            "source_1", key
        ): value
        for key, value in credentials.items()
    }
    k8s_secrets_mock.assert_project_secrets(
        config.marketplace.k8s_secrets_project_name, expected_credentials
    )


def test_marketplace_source_manager(
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    manager = mlrun.api.crud.Marketplace()

    credentials = {"secret1": "value1", "secret2": "value2"}
    expected_credentials = {}
    for i in range(3):
        source_dict = _generate_source_dict(i, f"source_{i}", credentials)
        expected_credentials.update(
            {
                mlrun.api.crud.Marketplace()._generate_credentials_secret_key(
                    f"source_{i}", key
                ): value
                for key, value in credentials.items()
            }
        )
        source_object = mlrun.api.schemas.MarketplaceSource(**source_dict["source"])
        manager.add_source(source_object)

    k8s_secrets_mock.assert_project_secrets(
        config.marketplace.k8s_secrets_project_name, expected_credentials
    )

    manager.remove_source("source_1")
    for key in credentials:
        expected_credentials.pop(
            mlrun.api.crud.Marketplace()._generate_credentials_secret_key(
                "source_1", key
            )
        )
    k8s_secrets_mock.assert_project_secrets(
        config.marketplace.k8s_secrets_project_name, expected_credentials
    )

    # Test catalog retrieval, with various filters
    catalog = manager.get_source_catalog(source_object)
    assert len(catalog.catalog) == 5

    catalog = manager.get_source_catalog(source_object, channel="dev")
    assert len(catalog.catalog) == 1
    for item in catalog.catalog:
        assert item.metadata.name == "dev_function"

    catalog = manager.get_source_catalog(source_object, channel="prod")
    assert len(catalog.catalog) == 4
    for item in catalog.catalog:
        assert item.metadata.name in [
            "prod_function",
            "prod_function_2",
        ] and item.metadata.version in ["0.0.1", "1.0.0"]

    catalog = manager.get_source_catalog(source_object, channel="prod", version="1.0.0")
    assert len(catalog.catalog) == 2
    for item in catalog.catalog:
        assert (
            item.metadata.name in ["prod_function", "prod_function_2"]
            and item.metadata.version == "1.0.0"
        )

    item = manager.get_item(source_object, "prod_function", "prod", "1.0.0")
    assert (
        item.metadata.name == "prod_function"
        and item.metadata.version == "1.0.0"
        and item.metadata.channel == "prod"
    )


def test_marketplace_default_source(
    k8s_secrets_mock: tests.api.conftest.K8sSecretsMock,
) -> None:
    # This test validates that the default source is valid is its catalog and objects can be retrieved.
    manager = mlrun.api.crud.Marketplace()

    source_object = mlrun.api.schemas.MarketplaceSource.generate_default_source()
    catalog = manager.get_source_catalog(source_object)
    assert len(catalog.catalog) > 0
    print(f"Retrieved function catalog. Has {len(catalog.catalog)} functions in it.")
    # function = manager.get_item(source_object, "aggregate", "development", "0.0.1")
    for i in range(10):
        function = random.choice(catalog.catalog)
        print(
            f"Selected the following: function = {function.metadata.name}, channel = {function.metadata.channel},"
            + f" tag = {function.metadata.tag}, version = {function.metadata.version}"
        )

        function_yaml = manager.get_item_object_using_source_credentials(
            source_object, function.spec.item_uri + "src/function.yaml"
        )
        function_dict = yaml.safe_load(function_yaml)

        # Temporary fix, since there are some inconsistencies there - and _ are exchanged between the catalog.json
        # and the function.yaml
        yaml_function_name = function_dict["metadata"]["name"].replace("_", "-")
        function_modified_name = function.metadata.name.replace("_", "-")

        assert yaml_function_name == function_modified_name


def test_marketplace_catalog_apis(
    db: Session, client: TestClient, k8s_secrets_mock: tests.api.conftest.K8sSecretsMock
) -> None:
    # Get the global hub source-name
    sources = client.get("marketplace/sources").json()
    source_name = sources[0]["source"]["metadata"]["name"]

    catalog = client.get(f"marketplace/sources/{source_name}/items").json()
    item = random.choice(catalog["catalog"])
    url = item["spec"]["item_uri"] + "src/function.yaml"

    function_yaml = client.get(
        f"marketplace/sources/{source_name}/item-object", params={"url": url}
    )

    function_dict = yaml.safe_load(function_yaml.content)

    # Temporary fix, since there are some inconsistencies there - and _ are exchanged between the catalog.json
    # and the function.yaml
    yaml_function_name = function_dict["metadata"]["name"].replace("_", "-")
    function_modified_name = item["metadata"]["name"].replace("_", "-")

    assert yaml_function_name == function_modified_name
