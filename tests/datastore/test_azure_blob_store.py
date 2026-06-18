# Copyright 2025 Iguazio
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

from unittest.mock import Mock, patch

import pytest

import mlrun.errors
from mlrun.datastore.azure_blob import AzureBlobStore


class TestAzureBlobStore:
    """Unit tests for AzureBlobStore URL handling and Spark integration"""

    def setup_method(self):
        """Setup method to create a mock parent"""
        self.parent = Mock()
        # Configure parent mock to return None for any secret lookups by default
        self.parent.secret.return_value = None

    def _create_store(self, schema="az", endpoint="test-container", secrets=None):
        """Helper method to create AzureBlobStore instances"""
        return AzureBlobStore(
            parent=self.parent,
            schema=schema,
            name="test-store",
            endpoint=endpoint,
            secrets=secrets or {},
        )

    # --- get_read_only_https_url ---

    def _store_with_mock_client(
        self, secrets, hostname="acct.blob.core.windows.net", udk="udk"
    ):
        store = self._create_store(schema="az", endpoint="data", secrets=secrets)
        client = Mock()
        client.primary_hostname = hostname
        client.get_user_delegation_key.return_value = udk
        store._service_client = client
        return store, client

    def test_read_only_url_service_principal_user_delegation(self):
        store, client = self._store_with_mock_client({"account_name": "acct"})
        with patch(
            "mlrun.datastore.azure_blob.generate_blob_sas", return_value="sig=x"
        ) as gen:
            url = store.get_read_only_https_url("/projects/x/src.tar.gz")
        assert (
            url == "https://acct.blob.core.windows.net/data/projects/x/src.tar.gz?sig=x"
        )
        client.get_user_delegation_key.assert_called_once()
        kwargs = gen.call_args.kwargs
        assert kwargs["user_delegation_key"] == "udk"
        assert kwargs["account_name"] == "acct"
        assert kwargs["container_name"] == "data"
        assert kwargs["blob_name"] == "projects/x/src.tar.gz"
        assert kwargs["permission"].read is True
        assert kwargs["start"] < kwargs["expiry"]

    def test_read_only_url_account_key(self):
        store, client = self._store_with_mock_client(
            {"account_name": "acct", "account_key": "the-key"}
        )
        with patch(
            "mlrun.datastore.azure_blob.generate_blob_sas", return_value="sig=ak"
        ) as gen:
            url = store.get_read_only_https_url("/src.tar.gz")
        assert url == "https://acct.blob.core.windows.net/data/src.tar.gz?sig=ak"
        client.get_user_delegation_key.assert_not_called()
        assert gen.call_args.kwargs["account_key"] == "the-key"

    def test_read_only_url_sas_passthrough(self):
        store, client = self._store_with_mock_client(
            {"account_name": "acct", "sas_token": "?sv=2023&sig=abc"}
        )
        with patch("mlrun.datastore.azure_blob.generate_blob_sas") as gen:
            url = store.get_read_only_https_url("/src.tar.gz")
        assert (
            url == "https://acct.blob.core.windows.net/data/src.tar.gz?sv=2023&sig=abc"
        )
        gen.assert_not_called()
        client.get_user_delegation_key.assert_not_called()

    def test_read_only_url_connection_string_account_key(self):
        cs = (
            "DefaultEndpointsProtocol=https;AccountName=connacct;"
            "AccountKey=YWJjZA==;EndpointSuffix=core.windows.net"
        )
        store, client = self._store_with_mock_client(
            {"connection_string": cs}, hostname="connacct.blob.core.windows.net"
        )
        with patch(
            "mlrun.datastore.azure_blob.generate_blob_sas", return_value="sig=cs"
        ) as gen:
            url = store.get_read_only_https_url("/src.tar.gz")
        assert url == "https://connacct.blob.core.windows.net/data/src.tar.gz?sig=cs"
        kwargs = gen.call_args.kwargs
        assert kwargs["account_key"] == "YWJjZA=="
        assert kwargs["account_name"] == "connacct"
        client.get_user_delegation_key.assert_not_called()

    def test_read_only_url_encodes_blob_name(self):
        store, _ = self._store_with_mock_client(
            {"account_name": "acct", "account_key": "k"}
        )
        with patch(
            "mlrun.datastore.azure_blob.generate_blob_sas", return_value="sig=x"
        ) as gen:
            url = store.get_read_only_https_url("/projects/a b/src.tar.gz")
        assert (
            url
            == "https://acct.blob.core.windows.net/data/projects/a%20b/src.tar.gz?sig=x"
        )
        # SAS signs the raw blob name.
        assert gen.call_args.kwargs["blob_name"] == "projects/a b/src.tar.gz"

    def test_read_only_url_account_key_failure_not_mislabeled(self):
        store, _ = self._store_with_mock_client(
            {"account_name": "acct", "account_key": "k"}
        )
        with (
            patch(
                "mlrun.datastore.azure_blob.generate_blob_sas",
                side_effect=ValueError("bad key"),
            ),
            pytest.raises(mlrun.errors.MLRunRuntimeError) as exc_info,
        ):
            store.get_read_only_https_url("/src.tar.gz")
        assert "Storage Blob Delegator" not in str(exc_info.value)
        assert "bad key" in str(exc_info.value)

    def test_read_only_url_user_delegation_failure_hints_role(self):
        store, client = self._store_with_mock_client({"account_name": "acct"})
        client.get_user_delegation_key.side_effect = Exception("boom")
        with pytest.raises(
            mlrun.errors.MLRunRuntimeError, match="Storage Blob Delegator"
        ):
            store.get_read_only_https_url("/src.tar.gz")

    def test_read_only_url_no_container_raises(self):
        store = self._create_store(
            schema="az", endpoint="", secrets={"account_name": "acct"}
        )
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            store.get_read_only_https_url("/src.tar.gz")

    def test_read_only_url_missing_credentials_raises_client_error(self):
        # No credentials: the service_client build (outside the try) must surface as a
        # client-side MLRunInvalidArgumentError (400), not get wrapped as a 500.
        store = self._create_store(schema="az", endpoint="data", secrets={})
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            store.get_read_only_https_url("/src.tar.gz")

    def test_spark_url_az_schema_with_endpoint_container(self):
        """Test spark_url generation for az:// URLs where endpoint is container"""
        secrets = {
            "account_name": "teststorage",
            "account_key": "**",
        }
        store = self._create_store(schema="az", endpoint="mycontainer", secrets=secrets)

        result = store.spark_url
        expected = "wasbs://mycontainer@teststorage.blob.core.windows.net"
        assert result == expected

    def test_spark_url_az_schema_with_container_in_storage_options(self):
        """Test spark_url generation when container is in storage options"""
        store = self._create_store(schema="az", endpoint="someendpoint")

        mock_storage_options = {
            "container": "mycontainer",
            "account_name": "teststorage",
            "account_key": "**",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.spark_url
            expected = "wasbs://mycontainer@teststorage.blob.core.windows.net"
            assert result == expected

    def test_spark_url_wasbs_schema_with_container_in_storage_options(self):
        """Test spark_url generation for wasbs:// URLs"""
        store = self._create_store(
            schema="wasbs", endpoint="testdata.blob.core.windows.net"
        )

        mock_storage_options = {
            "container": "testcontainer",
            "account_name": "testdata",
            "account_key": "**",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.spark_url
            expected = "wasbs://testcontainer@testdata.blob.core.windows.net"
            assert result == expected

    def test_spark_url_wasbs_schema_with_endpoint_as_host(self):
        """Test that wasbs:// URLs use endpoint as host, not container"""
        store = self._create_store(
            schema="wasbs", endpoint="testdata.blob.core.windows.net"
        )

        mock_storage_options = {"container": "mycontainer"}
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.spark_url
            expected = "wasbs://mycontainer@testdata.blob.core.windows.net"
            assert result == expected

    def test_spark_url_connection_string_with_sas_token(self):
        """Test spark_url generation with SAS token connection string"""
        connection_string = (
            "BlobEndpoint=https://testdata.blob.core.windows.net/;"
            "QueueEndpoint=https://testdata.queue.core.windows.net/;"
            "FileEndpoint=https://testdata.file.core.windows.net/;"
            "TableEndpoint=https://testdata.table.core.windows.net/;"
            "SharedAccessSignature=**"
        )

        store = self._create_store(schema="az", endpoint="mycontainer")

        mock_storage_options = {
            "connection_string": connection_string,
            "container": "testcontainer",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.spark_url
            expected = "wasbs://testcontainer@testdata.blob.core.windows.net"
            assert result == expected

    def test_spark_url_connection_string_account_key(self):
        """Test spark_url generation with account key connection string"""
        connection_string = (
            "DefaultEndpointsProtocol=https;"
            "AccountName=testdata;"
            "AccountKey=**;"
            "EndpointSuffix=core.windows.net"
        )

        store = self._create_store(
            schema="wasbs", endpoint="testdata.blob.core.windows.net"
        )

        mock_storage_options = {
            "connection_string": connection_string,
            "container": "mycontainer",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.spark_url
            expected = "wasbs://mycontainer@testdata.blob.core.windows.net"
            assert result == expected

    def test_spark_url_no_container_raises_error(self):
        """Test that missing container raises appropriate error"""
        store = self._create_store(schema="az", endpoint="")  # Empty endpoint

        mock_storage_options = {
            "account_name": "teststorage",
            "account_key": "**",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match="Container name is required",
            ):
                _ = store.spark_url

    def test_spark_url_wasbs_no_container_raises_specific_error(self):
        """Test that wasbs:// URLs without container raise specific error"""
        store = self._create_store(
            schema="wasbs", endpoint="testdata.blob.core.windows.net"
        )

        mock_storage_options = {
            "account_name": "testdata",
            "account_key": "**",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match="Container name is required",
            ):
                _ = store.spark_url

    def test_spark_url_no_host_raises_error(self):
        """Test that missing host information raises appropriate error"""
        store = self._create_store(schema="az", endpoint="mycontainer")

        # Mock the secret method to return None for all keys
        with patch.object(store, "_get_secret_or_env", return_value=None):
            # This will force _storage_options to be empty after sanitization
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError, match="account_name is required"
            ):
                _ = store.spark_url

    def test_spark_url_wasbs_uses_endpoint_as_host(self):
        """Test that wasbs:// URLs can use endpoint as host when no other host info available"""
        store = self._create_store(
            schema="wasbs", endpoint="custom.blob.core.windows.net"
        )

        mock_storage_options = {"container": "testcontainer"}
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.spark_url
            expected = "wasbs://testcontainer@custom.blob.core.windows.net"
            assert result == expected

    def test_get_spark_options_with_sas_token(self):
        """Test Spark options generation with SAS token"""
        store = self._create_store(schema="az", endpoint="mycontainer")

        mock_storage_options = {
            "account_name": "teststorage",
            "sas_token": "**",
            "container": "mycontainer",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.get_spark_options()
            expected_key = "spark.hadoop.fs.azure.sas.mycontainer.teststorage.blob.core.windows.net"
            assert expected_key in result
            assert result[expected_key] == "**"

    def test_get_spark_options_with_account_key(self):
        """Test Spark options generation with account key"""
        store = self._create_store(schema="az", endpoint="mycontainer")

        mock_storage_options = {
            "account_name": "teststorage",
            "account_key": "**",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.get_spark_options()
            expected_key = (
                "spark.hadoop.fs.azure.account.key.teststorage.blob.core.windows.net"
            )
            assert expected_key in result
            assert result[expected_key] == "**"

    def test_get_spark_options_sas_no_container_raises_error(self):
        """Test that SAS token without container raises error"""
        store = self._create_store(schema="az", endpoint="")  # Empty endpoint

        mock_storage_options = {
            "account_name": "teststorage",
            "sas_token": "**",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match="Container name is required for WASB SAS",
            ):
                store.get_spark_options()

    @patch("mlrun.datastore.azure_blob.parse_connection_str")
    def test_get_spark_options_connection_string_parsing(self, mock_parse_conn):
        """Test Spark options generation with connection string parsing"""
        mock_parse_conn.return_value = (
            "testdata.blob.core.windows.net",
            None,
            {
                "account_name": "testdata",
                "sas_token": (
                    "sv=TESTVER&ss=TEST&srt=TEST&sp=TESTPERM"
                    "&se=2099-01-01T00:00:00Z&st=2020-01-01T00:00:00Z&spr=https&sig=DUMMYSIG"
                ),
            },
        )

        store = self._create_store(schema="az", endpoint="mycontainer")

        mock_storage_options = {
            "connection_string": "BlobEndpoint=https://testdata.blob.core.windows.net/;SharedAccessSignature=**",
            "container": "mycontainer",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.get_spark_options()
            expected_key = (
                "spark.hadoop.fs.azure.sas.mycontainer.testdata.blob.core.windows.net"
            )
            assert expected_key in result
            # Should have the parsed SAS token value
            expected_sas = (
                "sv=TESTVER&ss=TEST&srt=TEST&sp=TESTPERM"
                "&se=2099-01-01T00:00:00Z&st=2020-01-01T00:00:00Z&spr=https&sig=DUMMYSIG"
            )
            assert result[expected_key] == expected_sas

    def test_different_url_schemas_mapping(self):
        """Test that different schemas (az, wasbs, wasb) all create AzureBlobStore"""
        schemas = ["az", "wasbs", "wasb"]
        for schema in schemas:
            store = self._create_store(schema=schema, endpoint="test-endpoint")
            assert isinstance(store, AzureBlobStore)
            assert store.kind == schema

    def test_spark_url_priority_container_from_storage_options(self):
        """Test that storage_options container takes priority over endpoint"""
        store = self._create_store(schema="az", endpoint="endpoint-container")

        mock_storage_options = {
            "container": "storage-options-container",
            "account_name": "teststorage",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.spark_url
            expected = (
                "wasbs://storage-options-container@teststorage.blob.core.windows.net"
            )
            assert result == expected

    def test_spark_url_connection_string_host_priority(self):
        """Test that connection string host takes priority over account_name"""
        connection_string = "BlobEndpoint=https://priority.blob.core.windows.net/;"

        store = self._create_store(schema="az", endpoint="mycontainer")

        with patch("mlrun.datastore.azure_blob.parse_connection_str") as mock_parse:
            mock_parse.return_value = (
                "priority.blob.core.windows.net",
                None,
                {"account_name": "priority"},
            )

            mock_storage_options = {
                "connection_string": connection_string,
                "container": "mycontainer",
                "account_name": "fallback",  # Should be ignored in favor of connection string
            }
            with patch.object(store, "_storage_options", mock_storage_options):
                result = store.spark_url
                expected = "wasbs://mycontainer@priority.blob.core.windows.net"
                assert result == expected

    def test_spark_url_connection_string_blob_endpoint_only(self):
        """Test connection string with only BlobEndpoint (minimal format)"""
        connection_string = (
            "BlobEndpoint=https://teststorage.blob.core.windows.net/;"
            "AccountName=teststorage;"
            "AccountKey=**"
        )

        store = self._create_store(schema="az", endpoint="mycontainer")

        with patch("mlrun.datastore.azure_blob.parse_connection_str") as mock_parse:
            mock_parse.return_value = (
                "teststorage.blob.core.windows.net",
                None,
                {"account_name": "teststorage", "account_key": "**"},
            )

            mock_storage_options = {
                "connection_string": connection_string,
                "container": "mycontainer",
            }
            with patch.object(store, "_storage_options", mock_storage_options):
                result = store.spark_url
                expected = "wasbs://mycontainer@teststorage.blob.core.windows.net"
                assert result == expected

    def test_spark_url_connection_string_custom_domain(self):
        """Test connection string with custom domain"""
        connection_string = (
            "BlobEndpoint=https://mystorageaccount.mydomain.com/;"
            "AccountName=mystorageaccount;"
            "AccountKey=**"
        )

        store = self._create_store(schema="az", endpoint="mycontainer")

        with patch("mlrun.datastore.azure_blob.parse_connection_str") as mock_parse:
            mock_parse.return_value = (
                "mystorageaccount.mydomain.com",
                None,
                {"account_name": "mystorageaccount", "account_key": "**"},
            )

            mock_storage_options = {
                "connection_string": connection_string,
                "container": "mycontainer",
            }
            with patch.object(store, "_storage_options", mock_storage_options):
                result = store.spark_url
                expected = "wasbs://mycontainer@mystorageaccount.mydomain.com"
                assert result == expected

    def test_spark_url_connection_string_china_cloud(self):
        """Test connection string with China cloud endpoint suffix"""
        connection_string = (
            "DefaultEndpointsProtocol=https;"
            "AccountName=chinaaccount;"
            "AccountKey=**;"
            "EndpointSuffix=core.chinacloudapi.cn"
        )

        store = self._create_store(schema="az", endpoint="mycontainer")

        with patch("mlrun.datastore.azure_blob.parse_connection_str") as mock_parse:
            mock_parse.return_value = (
                "chinaaccount.blob.core.chinacloudapi.cn",
                None,
                {"account_name": "chinaaccount", "account_key": "**"},
            )

            mock_storage_options = {
                "connection_string": connection_string,
                "container": "mycontainer",
            }
            with patch.object(store, "_storage_options", mock_storage_options):
                result = store.spark_url
                expected = "wasbs://mycontainer@chinaaccount.blob.core.chinacloudapi.cn"
                assert result == expected

    def test_spark_url_malformed_connection_string_handling(self):
        """Test error handling for malformed connection strings"""
        store = self._create_store(schema="az", endpoint="mycontainer")

        # Mock a malformed connection string that causes parse_connection_str to raise
        with patch("mlrun.datastore.azure_blob.parse_connection_str") as mock_parse:
            mock_parse.side_effect = ValueError("Invalid connection string format")

            mock_storage_options = {
                "connection_string": "DefaultEndpointsProtocol=https;...and_some_bad_data;",
                "container": "mycontainer",
            }
            with patch.object(store, "_storage_options", mock_storage_options):
                # Should fallback gracefully when connection string parsing fails
                with pytest.raises(
                    mlrun.errors.MLRunInvalidArgumentError,
                    match="account_name is required",
                ):
                    _ = store.spark_url

    def test_spark_url_mixed_authentication_sources(self):
        """Test mixed authentication: connection_string provides endpoint, storage_options provides container"""
        connection_string = "BlobEndpoint=https://mixedauth.blob.core.windows.net/"

        store = self._create_store(schema="az", endpoint="endpoint-container")

        with patch("mlrun.datastore.azure_blob.parse_connection_str") as mock_parse:
            mock_parse.return_value = (
                "mixedauth.blob.core.windows.net",
                None,
                {"account_name": "mixedauth"},
            )

            mock_storage_options = {
                "connection_string": connection_string,
                "container": "storage-options-container",  # Container from storage_options
                "account_key": "**",  # Auth from storage_options
            }
            with patch.object(store, "_storage_options", mock_storage_options):
                result = store.spark_url
                # Should use container from storage_options and host from connection string
                expected = (
                    "wasbs://storage-options-container@mixedauth.blob.core.windows.net"
                )
                assert result == expected

    def test_spark_url_wasbs_container_from_endpoint_url(self):
        """Test that wasbs:// URLs can extract container from endpoint netloc"""
        store = self._create_store(
            schema="wasbs", endpoint="testcontainer123@testdata.blob.core.windows.net"
        )

        mock_storage_options = {
            "account_name": "testdata",
            "account_key": "**",
            "container": "testcontainer123",  # Container extracted from endpoint in constructor
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            result = store.spark_url
            expected = "wasbs://testcontainer123@testdata.blob.core.windows.net"
            assert result == expected

    def test_spark_url_wasbs_container_from_connection_string_blob_endpoint(self):
        """Test that wasbs:// URLs can extract container from BlobEndpoint in connection string"""
        secrets = {
            "connection_string": "BlobEndpoint=https://testdata.blob.core.windows.net/testcontainer123/;SharedAccessSignature=sv=TESTVER&ss=TEST&srt=TEST&sp=TESTPERM&se=2099-01-01T00:00:00Z&st=2020-01-01T00:00:00Z&spr=https&sig=DUMMYSIG",
        }
        store = self._create_store(
            schema="wasbs",
            endpoint="testdata.blob.core.windows.net",  # Just hostname
            secrets=secrets,
        )

        result = store.spark_url
        expected = "wasbs://testcontainer123@testdata.blob.core.windows.net"
        assert result == expected

    def test_get_spark_options_wasbs_container_from_connection_string_blob_endpoint(
        self,
    ):
        """Test that get_spark_options can extract container from BlobEndpoint in connection string"""
        secrets = {
            "connection_string": "BlobEndpoint=https://testdata.blob.core.windows.net/testcontainer123/;SharedAccessSignature=sv=TESTVER&ss=TEST&srt=TEST&sp=TESTPERM&se=2099-01-01T00:00:00Z&st=2020-01-01T00:00:00Z&spr=https&sig=DUMMYSIG",
        }
        store = self._create_store(
            schema="wasbs",
            endpoint="testdata.blob.core.windows.net",  # Just hostname
            secrets=secrets,
        )

        result = store.get_spark_options()
        expected_key = (
            "spark.hadoop.fs.azure.sas.testcontainer123.testdata.blob.core.windows.net"
        )
        assert expected_key in result
        # Should have the parsed SAS token value
        expected_sas = (
            "sv=TESTVER&ss=TEST&srt=TEST&sp=TESTPERM"
            "&se=2099-01-01T00:00:00Z&st=2020-01-01T00:00:00Z&spr=https&sig=DUMMYSIG"
        )
        assert result[expected_key] == expected_sas

    def test_constructor_extracts_container_from_wasbs_endpoint(self):
        """Test that constructor extracts container from WASBS endpoint format"""
        # Test WASBS endpoint with container@host format
        store = self._create_store(
            schema="wasbs", endpoint="testcontainer123@testdata.blob.core.windows.net"
        )

        # Container should be extracted and endpoint should be cleaned
        assert store._container_from_endpoint == "testcontainer123"
        assert store.endpoint == "testdata.blob.core.windows.net"

    def test_storage_options_uses_container_from_constructor(self):
        """Test that storage_options property uses container extracted in constructor"""
        store = self._create_store(
            schema="wasbs",
            endpoint="testcontainer123@testdata.blob.core.windows.net",  # WASBS format
        )

        mock_secrets = {
            "connection_string": (
                "DefaultEndpointsProtocol=https;"
                "AccountName=testdata;"
                "AccountKey=**;"
                "EndpointSuffix=core.windows.net"
            )
        }

        with patch.object(store, "_get_secret_or_env") as mock_get_secret:

            def side_effect(key):
                return mock_secrets.get(key)

            mock_get_secret.side_effect = side_effect

            # Access storage_options to check container is included
            options = store.storage_options

            # Should include container from constructor
            assert options.get("container") == "testcontainer123"
            assert store.endpoint == "testdata.blob.core.windows.net"

    def test_full_wasbs_url_parsing_flow(self):
        """Test the complete flow from WASBS URL to working spark_url"""
        # This simulates the actual flow from get_or_create_store
        from mlrun.datastore.utils import parse_url

        # Step 1: parse_url should preserve container in netloc
        original_url = "wasbs://testcontainer123@testdata.blob.core.windows.net/testpath.example.com/test/"
        schema, endpoint, _ = parse_url(original_url)

        assert schema == "wasbs"
        assert (
            endpoint == "testcontainer123@testdata.blob.core.windows.net"
        )  # Should preserve container

        # Step 2: Create store - constructor should extract container automatically
        store = self._create_store(schema="wasbs", endpoint=endpoint)

        mock_secrets = {
            "connection_string": (
                "DefaultEndpointsProtocol=https;"
                "AccountName=testdata;"
                "AccountKey=**;"
                "EndpointSuffix=core.windows.net"
            )
        }

        with patch.object(store, "_get_secret_or_env") as mock_get_secret:

            def side_effect(key):
                return mock_secrets.get(key)

            mock_get_secret.side_effect = side_effect

            # Should work without raising an error
            spark_url = store.spark_url
            expected = "wasbs://testcontainer123@testdata.blob.core.windows.net"
            assert spark_url == expected

    def test_wasbs_url_parsing_preserves_container(self):
        """Test that parse_url correctly preserves container for WASBS URLs"""
        from mlrun.datastore.utils import parse_url

        test_cases = [
            {
                "url": "wasbs://container@host.blob.core.windows.net/path",
                "expected_schema": "wasbs",
                "expected_endpoint": "container@host.blob.core.windows.net",
                "expected_netloc": "container@host.blob.core.windows.net",
            },
            {
                "url": "wasb://mycontainer@myaccount.blob.core.windows.net/folder/file",
                "expected_schema": "wasb",
                "expected_endpoint": "mycontainer@myaccount.blob.core.windows.net",
                "expected_netloc": "mycontainer@myaccount.blob.core.windows.net",
            },
            {
                "url": "https://myaccount.blob.core.windows.net/container",  # Non-WASBS should work normally
                "expected_schema": "https",
                "expected_endpoint": "myaccount.blob.core.windows.net",  # Should be just hostname
                "expected_netloc": "myaccount.blob.core.windows.net",
            },
        ]

        for case in test_cases:
            schema, endpoint, parsed_url = parse_url(case["url"])
            assert schema == case["expected_schema"], (
                f"Schema mismatch for {case['url']}"
            )
            assert endpoint == case["expected_endpoint"], (
                f"Endpoint mismatch for {case['url']}"
            )
            assert parsed_url.netloc == case["expected_netloc"], (
                f"Netloc mismatch for {case['url']}"
            )

    def test_convert_key_to_remote_path_wasbs(self):
        """Test _convert_key_to_remote_path for WASBS URLs uses container, not hostname"""
        # For WASBS URLs, the endpoint after constructor is the hostname
        # but _convert_key_to_remote_path should use the container from storage_options
        connection_string = (
            "DefaultEndpointsProtocol=https;AccountName=testdata;"
            "AccountKey=xxx;EndpointSuffix=core.windows.net"
        )
        store = self._create_store(
            schema="wasbs",
            endpoint="testdata.blob.core.windows.net",  # hostname after container extraction
            secrets={"connection_string": connection_string},
        )
        store._container_from_endpoint = "testcbs"

        # Mock storage_options to include the container
        mock_storage_options = {
            "container": "testcbs",
            "account_name": "testdata",
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            # Test path conversion - should NOT include hostname
            result = store._convert_key_to_remote_path(
                "vmdev213.lab.iguazeng.com/cmtayiymak/1759222142345_752"
            )
            expected = "testcbs/vmdev213.lab.iguazeng.com/cmtayiymak/1759222142345_752"
            assert result == expected, f"Expected {expected}, got {result}"

            # Verify it does NOT include hostname
            assert "testdata.blob.core.windows.net" not in result

    def test_convert_key_to_remote_path_az(self):
        """Test _convert_key_to_remote_path for az:// URLs uses endpoint as container"""
        # For az:// URLs, endpoint IS the container name
        store = self._create_store(
            schema="az", endpoint="mycontainer", secrets={"account_name": "teststorage"}
        )

        result = store._convert_key_to_remote_path("path/to/file.txt")
        expected = "mycontainer/path/to/file.txt"
        assert result == expected

    def test_convert_key_to_remote_path_with_schema(self):
        """Test _convert_key_to_remote_path when key already has schema"""
        store = self._create_store(
            schema="wasbs",
            endpoint="testdata.blob.core.windows.net",
        )

        # When key has a schema, it should be returned as-is (stripped of leading /)
        result = store._convert_key_to_remote_path(
            "wasbs://container@host/path/to/file"
        )
        expected = "wasbs://container@host/path/to/file"
        assert result == expected

    def test_convert_key_to_remote_path_wasbs_no_container_fallback(self):
        """Test _convert_key_to_remote_path falls back to endpoint when container is missing for wasbs"""
        # Edge case: wasbs without container (invalid config, but should not crash)
        store = self._create_store(
            schema="wasbs",
            endpoint="testdata.blob.core.windows.net",
            secrets={"account_name": "testdata", "account_key": "xxx"},
        )

        # Mock storage_options WITHOUT container
        mock_storage_options = {
            "account_name": "testdata",
            "account_key": "xxx",
            # No container!
        }
        with patch.object(store, "_storage_options", mock_storage_options):
            # Should fall back to using endpoint (even though it's hostname)
            result = store._convert_key_to_remote_path("path/to/file.txt")
            # Falls back to endpoint (hostname) - not ideal but maintains backward compatibility
            expected = "testdata.blob.core.windows.net/path/to/file.txt"
            assert result == expected

    @pytest.mark.parametrize(
        "env_vars",
        [
            {
                "AZURE_STORAGE_ACCOUNT_NAME": "teststorage",
                "AZURE_STORAGE_ACCOUNT_KEY": "mlrun-key",
                "AZURE_STORAGE_CLIENT_ID": "storage-client-id",
                "AZURE_STORAGE_CLIENT_SECRET": "storage-client-secret",
                "AZURE_STORAGE_TENANT_ID": "storage-tenant-id",
            },
            {
                "AZURE_STORAGE_ACCOUNT": "teststorage",
                "AZURE_STORAGE_ACCESS_KEY": "sdk-key",
                "AZURE_CLIENT_ID": "sdk-client-id",
                "AZURE_CLIENT_SECRET": "sdk-client-secret",
                "AZURE_TENANT_ID": "sdk-tenant-id",
            },
        ],
        ids=["mlrun_env_vars", "standard_azure_sdk_env_vars"],
    )
    def test_storage_options_resolves_azure_credential_env_vars(self, env_vars):
        """Test that storage_options picks up both AZURE_STORAGE_* and standard Azure SDK env var names."""
        store = self._create_store(schema="az", endpoint="mycontainer")

        with patch.object(store, "_get_secret_or_env") as mock_get_secret:
            mock_get_secret.side_effect = lambda key: env_vars.get(key)

            options = store.storage_options

        assert options["account_name"] == "teststorage"
        assert options["account_key"] is not None
        assert options["client_id"] is not None
        assert options["client_secret"] is not None
        assert options["tenant_id"] is not None

    def test_storage_options_workload_identity_drops_partial_triple(self):
        """ML-12668: a client_id without a secret (workload identity) is dropped and anon set False."""
        env_vars = {
            "AZURE_STORAGE_ACCOUNT_NAME": "teststorage",
            "AZURE_CLIENT_ID": "wi-client-id",
            "AZURE_TENANT_ID": "wi-tenant-id",
            # no client secret anywhere — the webhook injects only id/tenant/federated-token-file
        }
        store = self._create_store(schema="az", endpoint="mycontainer")

        with patch.object(store, "_get_secret_or_env") as mock_get_secret:
            mock_get_secret.side_effect = lambda key: env_vars.get(key)
            options = store.storage_options

        assert "client_id" not in options
        assert "tenant_id" not in options
        assert options["anon"] is False
        assert options["account_name"] == "teststorage"

    def test_storage_options_keeps_full_service_principal(self):
        """A full service principal (client_id + client_secret + tenant_id) is preserved, anon untouched."""
        env_vars = {
            "AZURE_STORAGE_ACCOUNT_NAME": "teststorage",
            "AZURE_CLIENT_ID": "sp-client-id",
            "AZURE_CLIENT_SECRET": "sp-client-secret",
            "AZURE_TENANT_ID": "sp-tenant-id",
        }
        store = self._create_store(schema="az", endpoint="mycontainer")

        with patch.object(store, "_get_secret_or_env") as mock_get_secret:
            mock_get_secret.side_effect = lambda key: env_vars.get(key)
            options = store.storage_options

        assert options["client_id"] == "sp-client-id"
        assert options["client_secret"] == "sp-client-secret"
        assert options["tenant_id"] == "sp-tenant-id"
        assert "anon" not in options

    def test_storage_options_account_key_only_unchanged(self):
        """Account-key auth has no client_id, so the WI branch must not fire — no anon key added."""
        env_vars = {
            "AZURE_STORAGE_ACCOUNT_NAME": "teststorage",
            "AZURE_STORAGE_ACCOUNT_KEY": "the-key",
        }
        store = self._create_store(schema="az", endpoint="mycontainer")

        with patch.object(store, "_get_secret_or_env") as mock_get_secret:
            mock_get_secret.side_effect = lambda key: env_vars.get(key)
            options = store.storage_options

        assert options["account_key"] == "the-key"
        assert "client_id" not in options
        assert "anon" not in options

    def test_do_connect_workload_identity_uses_default_credential(self):
        """With anon=False and no key/SAS/client_id, _do_connect uses DefaultAzureCredential."""
        store = self._create_store(schema="az", endpoint="mycontainer")
        mock_storage_options = {"account_name": "teststorage", "anon": False}

        with (
            patch.object(store, "_storage_options", mock_storage_options),
            patch("azure.identity.DefaultAzureCredential") as mock_default,
            patch("azure.identity.ClientSecretCredential") as mock_client_secret,
            patch("mlrun.datastore.azure_blob.BlobServiceClient") as mock_blob_client,
        ):
            store._do_connect()

        mock_default.assert_called_once()
        mock_client_secret.assert_not_called()
        _, kwargs = mock_blob_client.call_args
        assert kwargs["credential"] is mock_default.return_value

    def test_do_connect_service_principal_uses_client_secret_credential(self):
        """A full service principal still builds a ClientSecretCredential with the supplied secret."""
        store = self._create_store(schema="az", endpoint="mycontainer")
        mock_storage_options = {
            "account_name": "teststorage",
            "client_id": "sp-client-id",
            "client_secret": "sp-client-secret",
            "tenant_id": "sp-tenant-id",
        }

        with (
            patch.object(store, "_storage_options", mock_storage_options),
            patch("azure.identity.DefaultAzureCredential") as mock_default,
            patch("azure.identity.ClientSecretCredential") as mock_client_secret,
            patch("mlrun.datastore.azure_blob.BlobServiceClient") as mock_blob_client,
        ):
            store._do_connect()

        mock_client_secret.assert_called_once_with(
            tenant_id="sp-tenant-id",
            client_id="sp-client-id",
            client_secret="sp-client-secret",
        )
        mock_default.assert_not_called()
        _, kwargs = mock_blob_client.call_args
        assert kwargs["credential"] is mock_client_secret.return_value

    def test_filesystem_missing_account_name_and_connection_string_raises(self):
        """ML-12692: az:// with neither account_name nor connection_string fails fast.

        This is the Azure identity path: credentials are routed via anon=False, but the
        storage account name was never supplied. Instead of letting adlfs raise its cryptic
        'Must provide ... account_name with credentials', we raise a clear MLRun error.
        """
        store = self._create_store(schema="az", endpoint="data")
        mock_storage_options = {"anon": False, "container": "data"}

        with patch.object(store, "_storage_options", mock_storage_options):
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError, match="account_name"
            ):
                _ = store.filesystem

    def test_filesystem_with_account_name_does_not_raise(self):
        """When account_name is present, the account-name guard does not trigger."""
        store = self._create_store(schema="az", endpoint="data")
        mock_storage_options = {"account_name": "teststorage", "anon": False}

        with (
            patch.object(store, "_storage_options", mock_storage_options),
            patch("mlrun.datastore.azure_blob.get_filesystem_class") as mock_get_class,
            patch(
                "mlrun.datastore.azure_blob.make_datastore_schema_sanitizer"
            ) as mock_make_fs,
        ):
            result = store.filesystem

        mock_get_class.assert_called_once()
        mock_make_fs.assert_called_once()
        assert result is mock_make_fs.return_value
