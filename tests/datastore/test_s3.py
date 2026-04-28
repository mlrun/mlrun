# Copyright 2026 Iguazio
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

import botocore.exceptions
import pytest

from mlrun.datastore.s3 import S3Store


class TestS3StoreExceptionHandling:
    """Unit tests for S3Store exception handling logic.

    Tests specifically cover the exception handling in the S3Store.get() method,
    focusing on the conversion of NoSuchKey errors to FileNotFoundError.
    """

    @pytest.fixture
    @patch("boto3.resource")
    @patch("mlrun.datastore.s3.DataStore.__init__")
    @patch("mlrun.datastore.s3.S3Store._get_secret_or_env")
    def s3_store(
        self,
        mock_get_secret_or_env: Mock,
        mock_datastore_init: Mock,
        mock_boto3_resource: Mock,
    ) -> S3Store:
        """Create a real S3Store instance with mocked boto3 dependencies."""
        mock_datastore_init.return_value = None
        mock_get_secret_or_env.return_value = None  # No environment secrets

        # Create real S3Store instance
        s3_store = S3Store(
            parent=None, schema="s3", name="test", endpoint="test-bucket", secrets=None
        )

        # Manually set up required attributes that DataStore.__init__ would normally set
        s3_store._secrets = {}
        s3_store._parent = None
        s3_store.kind = "s3"
        s3_store.name = "test"
        s3_store.endpoint = "test-bucket"
        s3_store._filesystem = None

        # Mock required methods that might be called
        s3_store._get_parent_secret = Mock(return_value=None)
        s3_store._join = Mock(side_effect=lambda key: f"/{key}")
        s3_store._prepare_put_data = Mock(return_value=(b"data", None))
        s3_store._sanitize_options = Mock(side_effect=lambda x: x)

        # Mock the boto3 resource
        s3_store.s3 = mock_boto3_resource.return_value

        return s3_store

    @pytest.fixture
    def mock_s3_obj(self, s3_store: S3Store) -> Mock:
        """Create a mock S3 object for testing."""
        mock_obj = Mock()
        s3_store.s3.Object.return_value = mock_obj
        return mock_obj

    def test_get_handles_nosuchkey_error_raises_filenotfound(
        self, s3_store: S3Store, mock_s3_obj: Mock
    ) -> None:
        """Test that NoSuchKey error is converted to FileNotFoundError.

        Verifies that when AWS S3 returns a NoSuchKey error, the S3Store.get() method
        correctly converts it to a FileNotFoundError with proper S3 URI format and
        preserves the original exception as the cause.
        """
        # Create a mock ClientError with NoSuchKey error code
        error_response = {
            "Error": {
                "Code": "NoSuchKey",
                "Message": "The specified key does not exist.",
            }
        }
        client_error = botocore.exceptions.ClientError(
            error_response=error_response, operation_name="GetObject"
        )

        # Mock the object.get() method to raise the ClientError
        mock_s3_obj.get.side_effect = client_error

        # Test that FileNotFoundError is raised with correct message
        with pytest.raises(FileNotFoundError) as exc_info:
            s3_store.get("test-key")

        assert str(exc_info.value) == "s3://test-bucket/test-key"
        assert exc_info.value.__cause__ is client_error

    def test_get_handles_nosuchkey_error_with_size_and_offset_raises_filenotfound(
        self, s3_store: S3Store, mock_s3_obj: Mock
    ) -> None:
        """Test that NoSuchKey error is converted to FileNotFoundError when using range requests.

        Verifies that partial read requests (with size and offset) also properly handle
        NoSuchKey errors and that the Range header is correctly set.
        """
        # Create a mock ClientError with NoSuchKey error code
        error_response = {
            "Error": {
                "Code": "NoSuchKey",
                "Message": "The specified key does not exist.",
            }
        }
        client_error = botocore.exceptions.ClientError(
            error_response=error_response, operation_name="GetObject"
        )

        # Mock the object.get() method to raise the ClientError
        mock_s3_obj.get.side_effect = client_error

        # Test that FileNotFoundError is raised with size and offset parameters
        with pytest.raises(FileNotFoundError) as exc_info:
            s3_store.get("test-key", size=100, offset=50)

        assert str(exc_info.value) == "s3://test-bucket/test-key"
        assert exc_info.value.__cause__ is client_error

        # Verify that get() was called with Range parameter
        mock_s3_obj.get.assert_called_once_with(Range="bytes=50-149")

    def test_get_reraises_other_client_errors(
        self, s3_store: S3Store, mock_s3_obj: Mock
    ) -> None:
        """Test that non-NoSuchKey ClientErrors are re-raised as-is.

        Ensures that other AWS errors (like AccessDenied) are not converted to
        FileNotFoundError but are passed through unchanged.
        """
        # Create a mock ClientError with a different error code
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}
        client_error = botocore.exceptions.ClientError(
            error_response=error_response, operation_name="GetObject"
        )

        # Mock the object.get() method to raise the ClientError
        mock_s3_obj.get.side_effect = client_error

        # Test that the original ClientError is re-raised
        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            s3_store.get("test-key")

        assert exc_info.value is client_error

    def test_get_success_without_size_offset(
        self, s3_store: S3Store, mock_s3_obj: Mock
    ) -> None:
        """Test successful get operation without size and offset.

        Verifies the happy path where S3 returns data successfully for a full object read.
        """
        # Mock successful response
        mock_body = Mock()
        mock_body.read.return_value = b"test data"
        mock_s3_obj.get.return_value = {"Body": mock_body}

        result = s3_store.get("test-key")

        assert result == b"test data"
        mock_s3_obj.get.assert_called_once_with()

    def test_get_success_with_size_and_offset(
        self, s3_store: S3Store, mock_s3_obj: Mock
    ) -> None:
        """Test successful get operation with size and offset.

        Verifies that partial read requests work correctly and generate proper Range headers.
        """
        # Mock successful response
        mock_body = Mock()
        mock_body.read.return_value = b"partial data"
        mock_s3_obj.get.return_value = {"Body": mock_body}

        result = s3_store.get("test-key", size=100, offset=50)

        assert result == b"partial data"
        mock_s3_obj.get.assert_called_once_with(Range="bytes=50-149")

    def test_get_success_with_offset_only(
        self, s3_store: S3Store, mock_s3_obj: Mock
    ) -> None:
        """Test successful get operation with offset only.

        Verifies that reads from a specific offset to end-of-file work correctly.
        """
        # Mock successful response
        mock_body = Mock()
        mock_body.read.return_value = b"data from offset"
        mock_s3_obj.get.return_value = {"Body": mock_body}

        result = s3_store.get("test-key", offset=100)

        assert result == b"data from offset"
        mock_s3_obj.get.assert_called_once_with(Range="bytes=100-")

    def test_get_range_static_method(self) -> None:
        """Test the static get_range method used by get().

        Verifies that the Range header generation logic works correctly for various
        combinations of size and offset parameters.
        """
        # Test with size and offset
        range_header = S3Store.get_range(100, 50)
        assert range_header == "bytes=50-149"

        # Test with offset only (size=None)
        range_header = S3Store.get_range(None, 100)
        assert range_header == "bytes=100-"

        # Test with size=0 (should be treated as no size)
        range_header = S3Store.get_range(0, 50)
        assert range_header == "bytes=50-"


class TestS3StoreAnonymousAccessFallback:
    """Tests for ML-11829: S3Store should use the AWS default credential chain
    (IAM roles, IRSA, etc.) when no explicit credentials are provided, and only
    fall back to anonymous access when no credentials are available at all.
    """

    @patch("mlrun.datastore.s3.S3Store._has_default_credentials", return_value=True)
    @patch("boto3.resource")
    @patch("mlrun.datastore.s3.DataStore.__init__", return_value=None)
    @patch("mlrun.datastore.s3.S3Store._get_secret_or_env", return_value=None)
    def test_no_anonymous_when_default_credentials_available(
        self,
        mock_get_secret: Mock,
        mock_init: Mock,
        mock_boto3_resource: Mock,
        mock_has_creds: Mock,
    ):
        """When IAM role or other default credentials exist, request signing stays enabled."""
        mock_resource = Mock()
        mock_boto3_resource.return_value = mock_resource

        S3Store(parent=None, schema="s3", name="test", endpoint="bucket")

        mock_resource.meta.client.meta.events.register.assert_not_called()

    @patch("mlrun.datastore.s3.S3Store._has_default_credentials", return_value=False)
    @patch("boto3.resource")
    @patch("mlrun.datastore.s3.DataStore.__init__", return_value=None)
    @patch("mlrun.datastore.s3.S3Store._get_secret_or_env", return_value=None)
    def test_anonymous_when_no_credentials_available(
        self,
        mock_get_secret: Mock,
        mock_init: Mock,
        mock_boto3_resource: Mock,
        mock_has_creds: Mock,
    ):
        """When no credentials are available at all, falls back to anonymous access."""
        mock_resource = Mock()
        mock_boto3_resource.return_value = mock_resource

        S3Store(parent=None, schema="s3", name="test", endpoint="bucket")

        mock_resource.meta.client.meta.events.register.assert_called_once()
        assert (
            mock_resource.meta.client.meta.events.register.call_args[0][0]
            == "choose-signer.s3.*"
        )
