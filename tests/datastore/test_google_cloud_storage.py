# Copyright 2024 Iguazio
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

from unittest.mock import MagicMock

import pytest

import mlrun.errors
from mlrun.datastore.google_cloud_storage import GoogleCloudStorageStore


def test_get_storage_options():
    st = GoogleCloudStorageStore(parent="parent", schema="schema", name="name")

    st._get_secret_or_env = MagicMock(return_value=None)
    use_listings_cache_dict = {"use_listings_cache": False}
    assert st.get_storage_options() == {**use_listings_cache_dict}

    st = GoogleCloudStorageStore(parent="parent", schema="schema", name="name")
    st._get_secret_or_env = MagicMock(
        return_value='{"key1": "value1", "key2": "value2"}'
    )
    assert st.get_storage_options() == {
        "token": {"key1": "value1", "key2": "value2"},
        **use_listings_cache_dict,
    }

    st = GoogleCloudStorageStore(parent="parent", schema="schema", name="name")
    st._get_secret_or_env = MagicMock(return_value="/path/to/gcs_credentials_file")
    assert st.get_storage_options() == {
        "token": "/path/to/gcs_credentials_file",
        **use_listings_cache_dict,
    }

    st = GoogleCloudStorageStore(parent="parent", schema="schema", name="name")
    st._get_secret_or_env = MagicMock(
        return_value={"token": {"key1": "value1", "key2": "value2"}}
    )
    assert st.get_storage_options() == {
        "token": {"token": {"key1": "value1", "key2": "value2"}},
        **use_listings_cache_dict,
    }


def _store_with_mock_client(endpoint="data", signed_url="https://signed"):
    store = GoogleCloudStorageStore(
        parent="parent", schema="gcs", name="name", endpoint=endpoint
    )
    blob = MagicMock()
    blob.generate_signed_url.return_value = signed_url
    client = MagicMock()
    client.bucket.return_value.blob.return_value = blob
    store._storage_client = client
    return store, client, blob


def test_read_only_url_signs_blob():
    store, client, blob = _store_with_mock_client()
    url = store.get_read_only_https_url("/projects/x/src.tar.gz")
    assert url == "https://signed"
    client.bucket.assert_called_once_with("data")
    client.bucket.return_value.blob.assert_called_once_with("projects/x/src.tar.gz")
    kwargs = blob.generate_signed_url.call_args.kwargs
    assert kwargs["version"] == "v4"
    assert kwargs["method"] == "GET"


def test_read_only_url_no_bucket_raises():
    store, _, _ = _store_with_mock_client(endpoint="")
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        store.get_read_only_https_url("/src.tar.gz")


def test_read_only_url_signing_failure_wrapped():
    store, _, blob = _store_with_mock_client()
    blob.generate_signed_url.side_effect = Exception("no signer")
    with pytest.raises(
        mlrun.errors.MLRunRuntimeError,
        match="failed to create read-only GCS signed URL",
    ) as exc_info:
        store.get_read_only_https_url("/src.tar.gz")
    assert "no signer" in str(exc_info.value)


def test_read_only_url_missing_credentials_raises_client_error():
    # Missing signing credentials should be a client-side error.
    store = GoogleCloudStorageStore(
        parent="parent", schema="gcs", name="name", endpoint="data"
    )
    store._get_secret_or_env = MagicMock(return_value=None)
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError, match="GCP_CREDENTIALS"):
        store.get_read_only_https_url("/src.tar.gz")
