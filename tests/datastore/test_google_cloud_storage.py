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
#

from unittest.mock import MagicMock

from mlrun.datastore.google_cloud_storage import GoogleCloudStorageStore


def test_get_storage_options():
    st = GoogleCloudStorageStore(parent="parent", schema="schema", name="name")

    st._get_secret_or_env = MagicMock(return_value=None)
    use_listings_cache_dict = {"use_listings_cache": False}
    assert st.get_storage_options() == {**use_listings_cache_dict}

    st._get_secret_or_env = MagicMock(
        return_value='{"key1": "value1", "key2": "value2"}'
    )
    assert st.get_storage_options() == {
        "token": {"key1": "value1", "key2": "value2"},
        **use_listings_cache_dict,
    }

    st._get_secret_or_env = MagicMock(return_value="/path/to/gcs_credentials_file")
    assert st.get_storage_options() == {
        "token": "/path/to/gcs_credentials_file",
        **use_listings_cache_dict,
    }

    st._get_secret_or_env = MagicMock(
        return_value={"token": {"key1": "value1", "key2": "value2"}}
    )
    assert st.get_storage_options() == {
        "token": {"token": {"key1": "value1", "key2": "value2"}},
        **use_listings_cache_dict,
    }
