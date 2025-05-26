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

import ast
import base64
import json
import typing
import warnings
from urllib.parse import ParseResult, urlparse

import pydantic.v1
from mergedeep import merge

import mlrun
import mlrun.errors

from ..secrets import get_secret_or_env


class DatastoreProfile(pydantic.v1.BaseModel):
    type: str
    name: str
    _private_attributes: list = ()

    class Config:
        extra = pydantic.v1.Extra.forbid

    @pydantic.v1.validator("name")
    @classmethod
    def lower_case(cls, v):
        return v.lower()

    @staticmethod
    def generate_secret_key(profile_name: str, project: str):
        secret_name_separator = "."
        full_key = (
            "datastore-profiles"
            + secret_name_separator
            + project
            + secret_name_separator
            + profile_name
        )
        return full_key

    def secrets(self) -> dict:
        return None

    def url(self, subpath) -> str:
        return None


class TemporaryClientDatastoreProfiles(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._data = {}  # Initialize the dictionary

    def add(self, profile: DatastoreProfile):
        self._data[profile.name] = profile

    def get(self, key):
        return self._data.get(key, None)

    def remove(self, key):
        self._data.pop(key, None)


class DatastoreProfileBasic(DatastoreProfile):
    type: str = pydantic.v1.Field("basic")
    _private_attributes = "private"
    public: str
    private: typing.Optional[str] = None


class ConfigProfile(DatastoreProfile):
    """
    A profile class for managing configuration data with nested public and private attributes.
    This class extends DatastoreProfile to handle configuration settings, separating them into
    public and private dictionaries. Both dictionaries support nested structures, and the class
    provides functionality to merge these attributes when needed.

    Args:
        public (Optional[dict]): Dictionary containing public configuration settings,
            supporting nested structures
        private (Optional[dict]): Dictionary containing private/sensitive configuration settings,
            supporting nested structures

    Example:
        >>> public = {
            "database": {
                "host": "localhost",
                "port": 5432
            },
            "api_version": "v1"
        }
        >>> private = {
            "database": {
                "password": "secret123",
                "username": "admin"
            },
            "api_key": "xyz789"
        }
        >>> config = ConfigProfile("myconfig", public=public, private=private)

        # When attributes() is called, it merges public and private:
        # {
        #     "database": {
        #         "host": "localhost",
        #         "port": 5432,
        #         "password": "secret123",
        #         "username": "admin"
        #     },
        #     "api_version": "v1",
        #     "api_key": "xyz789"
        # }

    """

    type = "config"
    _private_attributes = "private"
    public: typing.Optional[dict] = None
    private: typing.Optional[dict] = None

    def attributes(self):
        res = {}
        if self.public:
            res = merge(res, self.public)
        if self.private:
            res = merge(res, self.private)
        return res


class DatastoreProfileKafkaTarget(DatastoreProfile):
    type: str = pydantic.v1.Field("kafka_target")
    _private_attributes = "kwargs_private"
    bootstrap_servers: typing.Optional[str] = None
    brokers: typing.Optional[str] = None
    topic: str
    kwargs_public: typing.Optional[dict]
    kwargs_private: typing.Optional[dict]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.brokers and not self.bootstrap_servers:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "DatastoreProfileKafkaTarget requires the 'brokers' field to be set"
            )

        if self.bootstrap_servers:
            if self.brokers:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "DatastoreProfileKafkaTarget cannot be created with both 'brokers' and 'bootstrap_servers'"
                )
            else:
                self.brokers = self.bootstrap_servers
                self.bootstrap_servers = None
            warnings.warn(
                "'bootstrap_servers' parameter is deprecated in 1.7.0 and will be removed in 1.10.0, "
                "use 'brokers' instead.",
                # TODO: Remove this in 1.10.0
                FutureWarning,
            )

    def get_topic(self) -> typing.Optional[str]:
        return self.topic

    def attributes(self):
        attributes = {"brokers": self.brokers or self.bootstrap_servers}
        if self.kwargs_public:
            attributes = merge(attributes, self.kwargs_public)
        if self.kwargs_private:
            attributes = merge(attributes, self.kwargs_private)
        return attributes


class DatastoreProfileKafkaSource(DatastoreProfile):
    type: str = pydantic.v1.Field("kafka_source")
    _private_attributes = ("kwargs_private", "sasl_user", "sasl_pass")
    brokers: typing.Union[str, list[str]]
    topics: typing.Union[str, list[str]]
    group: typing.Optional[str] = "serving"
    initial_offset: typing.Optional[str] = "earliest"
    partitions: typing.Optional[typing.Union[str, list[str]]]
    sasl_user: typing.Optional[str]
    sasl_pass: typing.Optional[str]
    kwargs_public: typing.Optional[dict]
    kwargs_private: typing.Optional[dict]

    def get_topic(self) -> typing.Optional[str]:
        topics = [self.topics] if isinstance(self.topics, str) else self.topics
        return topics[0] if topics else None

    def attributes(self) -> dict[str, typing.Any]:
        attributes = {}
        if self.kwargs_public:
            attributes = merge(attributes, self.kwargs_public)
        if self.kwargs_private:
            attributes = merge(attributes, self.kwargs_private)

        topics = [self.topics] if isinstance(self.topics, str) else self.topics
        brokers = [self.brokers] if isinstance(self.brokers, str) else self.brokers

        attributes["brokers"] = brokers
        attributes["topics"] = topics
        attributes["group"] = self.group
        attributes["initial_offset"] = self.initial_offset
        if self.partitions is not None:
            attributes["partitions"] = self.partitions
        if sasl := mlrun.datastore.utils.KafkaParameters(attributes).sasl(
            usr=self.sasl_user, pwd=self.sasl_pass
        ):
            attributes["sasl"] = sasl
        return attributes


class DatastoreProfileV3io(DatastoreProfile):
    type: str = pydantic.v1.Field("v3io")
    v3io_access_key: typing.Optional[str] = None
    _private_attributes = "v3io_access_key"

    def url(self, subpath):
        subpath = subpath.lstrip("/")
        return f"v3io:///{subpath}"

    def secrets(self) -> dict:
        res = {}
        if self.v3io_access_key:
            res["V3IO_ACCESS_KEY"] = self.v3io_access_key
        return res


class DatastoreProfileS3(DatastoreProfile):
    type: str = pydantic.v1.Field("s3")
    _private_attributes = ("access_key_id", "secret_key")
    endpoint_url: typing.Optional[str] = None
    force_non_anonymous: typing.Optional[str] = None
    profile_name: typing.Optional[str] = None
    assume_role_arn: typing.Optional[str] = None
    access_key_id: typing.Optional[str] = None
    secret_key: typing.Optional[str] = None
    bucket: typing.Optional[str] = None

    @pydantic.v1.validator("bucket")
    @classmethod
    def check_bucket(cls, v):
        if not v:
            warnings.warn(
                "The 'bucket' attribute will be mandatory starting from version 1.10",
                FutureWarning,
                stacklevel=2,
            )
        return v

    def secrets(self) -> dict:
        res = {}
        if self.access_key_id:
            res["AWS_ACCESS_KEY_ID"] = self.access_key_id
        if self.secret_key:
            res["AWS_SECRET_ACCESS_KEY"] = self.secret_key
        if self.endpoint_url:
            res["S3_ENDPOINT_URL"] = self.endpoint_url
        if self.force_non_anonymous:
            res["S3_NON_ANONYMOUS"] = self.force_non_anonymous
        if self.profile_name:
            res["AWS_PROFILE"] = self.profile_name
        if self.assume_role_arn:
            res["MLRUN_AWS_ROLE_ARN"] = self.assume_role_arn
        return res

    def url(self, subpath):
        # TODO: There is an inconsistency with DatastoreProfileGCS. In DatastoreProfileGCS,
        # we assume that the subpath can begin without a '/' character,
        # while here we assume it always starts with one.
        if self.bucket:
            return f"s3://{self.bucket}{subpath}"
        else:
            return f"s3:/{subpath}"


class DatastoreProfileRedis(DatastoreProfile):
    type: str = pydantic.v1.Field("redis")
    _private_attributes = ("username", "password")
    endpoint_url: str
    username: typing.Optional[str] = None
    password: typing.Optional[str] = None

    def url_with_credentials(self):
        parsed_url = urlparse(self.endpoint_url)
        username = self.username
        password = self.password
        netloc = parsed_url.hostname
        if username:
            if password:
                netloc = f"{username}:{password}@{parsed_url.hostname}"
            else:
                netloc = f"{username}@{parsed_url.hostname}"

        if parsed_url.port:
            netloc += f":{parsed_url.port}"

        new_parsed_url = ParseResult(
            scheme=parsed_url.scheme,
            netloc=netloc,
            path=parsed_url.path,
            params=parsed_url.params,
            query=parsed_url.query,
            fragment=parsed_url.fragment,
        )
        return new_parsed_url.geturl()

    def secrets(self) -> dict:
        res = {}
        if self.username:
            res["REDIS_USER"] = self.username
        if self.password:
            res["REDIS_PASSWORD"] = self.password
        return res

    def url(self, subpath):
        return self.endpoint_url + subpath


class DatastoreProfileDBFS(DatastoreProfile):
    type: str = pydantic.v1.Field("dbfs")
    _private_attributes = ("token",)
    endpoint_url: typing.Optional[str] = None  # host
    token: typing.Optional[str] = None

    def url(self, subpath) -> str:
        return f"dbfs://{subpath}"

    def secrets(self) -> dict:
        res = {}
        if self.token:
            res["DATABRICKS_TOKEN"] = self.token
        if self.endpoint_url:
            res["DATABRICKS_HOST"] = self.endpoint_url
        return res


class DatastoreProfileGCS(DatastoreProfile):
    type: str = pydantic.v1.Field("gcs")
    _private_attributes = ("gcp_credentials",)
    credentials_path: typing.Optional[str] = None  # path to file.
    gcp_credentials: typing.Optional[typing.Union[str, dict]] = None
    bucket: typing.Optional[str] = None

    @pydantic.v1.validator("bucket")
    @classmethod
    def check_bucket(cls, v):
        if not v:
            warnings.warn(
                "The 'bucket' attribute will be mandatory starting from version 1.10",
                FutureWarning,
                stacklevel=2,
            )
        return v

    @pydantic.v1.validator("gcp_credentials", pre=True, always=True)
    @classmethod
    def convert_dict_to_json(cls, v):
        if isinstance(v, dict):
            return json.dumps(v)
        return v

    def url(self, subpath) -> str:
        # TODO: but there's something wrong with the subpath being assumed to not start with a slash here,
        # but the opposite assumption is made in S3.
        if subpath.startswith("/"):
            #  in gcs the path after schema is starts with bucket, wherefore it should not start with "/".
            subpath = subpath[1:]
        if self.bucket:
            return f"gcs://{self.bucket}/{subpath}"
        else:
            return f"gcs://{subpath}"

    def secrets(self) -> dict:
        res = {}
        if self.credentials_path:
            res["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        if self.gcp_credentials:
            res["GCP_CREDENTIALS"] = self.gcp_credentials
        return res


class DatastoreProfileAzureBlob(DatastoreProfile):
    type: str = pydantic.v1.Field("az")
    _private_attributes = (
        "connection_string",
        "account_key",
        "client_secret",
        "sas_token",
        "credential",
    )
    connection_string: typing.Optional[str] = None
    account_name: typing.Optional[str] = None
    account_key: typing.Optional[str] = None
    tenant_id: typing.Optional[str] = None
    client_id: typing.Optional[str] = None
    client_secret: typing.Optional[str] = None
    sas_token: typing.Optional[str] = None
    credential: typing.Optional[str] = None
    container: typing.Optional[str] = None

    @pydantic.v1.validator("container")
    @classmethod
    def check_container(cls, v):
        if not v:
            warnings.warn(
                "The 'container' attribute will be mandatory starting from version 1.10",
                FutureWarning,
                stacklevel=2,
            )
        return v

    def url(self, subpath) -> str:
        if subpath.startswith("/"):
            #  in azure the path after schema is starts with container, wherefore it should not start with "/".
            subpath = subpath[1:]
        if self.container:
            return f"az://{self.container}/{subpath}"
        else:
            return f"az://{subpath}"

    def secrets(self) -> dict:
        res = {}
        if self.connection_string:
            res["connection_string"] = self.connection_string
        if self.account_name:
            res["account_name"] = self.account_name
        if self.account_key:
            res["account_key"] = self.account_key
        if self.tenant_id:
            res["tenant_id"] = self.tenant_id
        if self.client_id:
            res["client_id"] = self.client_id
        if self.client_secret:
            res["client_secret"] = self.client_secret
        if self.sas_token:
            res["sas_token"] = self.sas_token
        if self.credential:
            res["credential"] = self.credential
        return res


class DatastoreProfileHdfs(DatastoreProfile):
    type: str = pydantic.v1.Field("hdfs")
    _private_attributes = "token"
    host: typing.Optional[str] = None
    port: typing.Optional[int] = None
    http_port: typing.Optional[int] = None
    user: typing.Optional[str] = None

    def secrets(self) -> dict:
        res = {}
        if self.host:
            res["HDFS_HOST"] = self.host
        if self.port:
            res["HDFS_PORT"] = self.port
        if self.port:
            res["HDFS_HTTP_PORT"] = self.http_port
        if self.user:
            res["HDFS_USER"] = self.user
        return res or None

    def url(self, subpath):
        return f"webhdfs://{self.host}:{self.http_port}{subpath}"


class DatastoreProfileTDEngine(DatastoreProfile):
    """
    A profile that holds the required parameters for a TDEngine database, with the websocket scheme.
    https://docs.tdengine.com/developer-guide/connecting-to-tdengine/#websocket-connection
    """

    type: str = pydantic.v1.Field("taosws")
    _private_attributes = ["password"]
    user: str
    # The password cannot be empty in real world scenarios. It's here just because of the profiles completion design.
    password: typing.Optional[str]
    host: str
    port: int

    def dsn(self) -> str:
        """Get the Data Source Name of the configured TDEngine profile."""
        return f"{self.type}://{self.user}:{self.password}@{self.host}:{self.port}"

    @classmethod
    def from_dsn(cls, dsn: str, profile_name: str) -> "DatastoreProfileTDEngine":
        """
        Construct a TDEngine profile from DSN (connection string) and a name for the profile.

        :param dsn:          The DSN (Data Source Name) of the TDEngine database, e.g.: ``"taosws://root:taosdata@localhost:6041"``.
        :param profile_name: The new profile's name.
        :return:             The TDEngine profile.
        """
        parsed_url = urlparse(dsn)
        return cls(
            name=profile_name,
            user=parsed_url.username,
            password=parsed_url.password,
            host=parsed_url.hostname,
            port=parsed_url.port,
        )


_DATASTORE_TYPE_TO_PROFILE_CLASS: dict[str, type[DatastoreProfile]] = {
    "v3io": DatastoreProfileV3io,
    "s3": DatastoreProfileS3,
    "redis": DatastoreProfileRedis,
    "basic": DatastoreProfileBasic,
    "kafka_target": DatastoreProfileKafkaTarget,
    "kafka_source": DatastoreProfileKafkaSource,
    "dbfs": DatastoreProfileDBFS,
    "gcs": DatastoreProfileGCS,
    "az": DatastoreProfileAzureBlob,
    "hdfs": DatastoreProfileHdfs,
    "taosws": DatastoreProfileTDEngine,
    "config": ConfigProfile,
}


class DatastoreProfile2Json(pydantic.v1.BaseModel):
    @staticmethod
    def _to_json(attributes):
        # First, base64 encode the values
        encoded_dict = {
            k: base64.b64encode(str(v).encode()).decode() for k, v in attributes.items()
        }
        # Then, return the dictionary as a JSON string with no spaces
        return json.dumps(encoded_dict).replace(" ", "")

    @staticmethod
    def get_json_public(profile: DatastoreProfile) -> str:
        return DatastoreProfile2Json._to_json(
            {
                k: v
                for k, v in profile.dict().items()
                if str(k) not in profile._private_attributes
            }
        )

    @staticmethod
    def get_json_private(profile: DatastoreProfile) -> str:
        return DatastoreProfile2Json._to_json(
            {
                k: v
                for k, v in profile.dict().items()
                if str(k) in profile._private_attributes
            }
        )

    @staticmethod
    def create_from_json(public_json: str, private_json: str = "{}"):
        attributes = json.loads(public_json)
        attributes_public = {
            k: base64.b64decode(str(v).encode()).decode() for k, v in attributes.items()
        }
        attributes = json.loads(private_json)
        attributes_private = {
            k: base64.b64decode(str(v).encode()).decode() for k, v in attributes.items()
        }
        decoded_dict = merge(attributes_public, attributes_private)

        def safe_literal_eval(value):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value

        decoded_dict = {k: safe_literal_eval(v) for k, v in decoded_dict.items()}
        datastore_type = decoded_dict.get("type")
        ds_profile_factory = _DATASTORE_TYPE_TO_PROFILE_CLASS
        if datastore_type in ds_profile_factory:
            return ds_profile_factory[datastore_type].parse_obj(decoded_dict)
        else:
            if datastore_type:
                reason = f"unexpected type '{decoded_dict['type']}'"
            else:
                reason = "missing type"
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Datastore profile failed to create from json due to {reason}"
            )


def datastore_profile_read(url, project_name="", secrets: typing.Optional[dict] = None):
    """
    Read and retrieve a datastore profile from a given URL.

    This function retrieves a datastore profile either from temporary client storage,
    or from the MLRun database. It handles both client-side and server-side profile formats
    and performs necessary conversions.

    Args:
        url (str): A URL with 'ds' scheme pointing to the datastore profile
            (e.g., 'ds://profile-name').
        project_name (str, optional): The project name where the profile is stored.
            Defaults to MLRun's default project.
        secrets (dict, optional): Dictionary containing secrets needed for profile retrieval.

    Returns:
        DatastoreProfile: The retrieved datastore profile object.

    Raises:
        MLRunInvalidArgumentError: In the following cases:
            - If the URL scheme is not 'ds'
            - If the profile cannot be retrieved from either server or local environment

    Note:
       When running from a client environment (outside MLRun pods), private profile information
       is not accessible. In this case, use register_temporary_client_datastore_profile() to
       register the profile with credentials for your local session. When running inside MLRun
       pods, the private information is automatically available and no temporary registration is needed.
    """

    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() != "ds":
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"resource URL '{url}' cannot be read as a datastore profile because its scheme is not 'ds'"
        )

    profile_name = parsed_url.hostname
    project_name = project_name or mlrun.mlconf.default_project
    datastore = TemporaryClientDatastoreProfiles().get(profile_name)
    if datastore:
        return datastore
    public_profile = mlrun.db.get_run_db().get_datastore_profile(
        profile_name, project_name
    )
    # The mlrun.db.get_run_db().get_datastore_profile() function is capable of returning
    # two distinct types of objects based on its execution context.
    # If it operates from the client or within the pod (which is the common scenario),
    # it yields an instance of `mlrun.datastore.DatastoreProfile`. Conversely,
    # when executed on the server with a direct call to `sqldb`, it produces an instance of
    # mlrun.common.schemas.DatastoreProfile.
    # In the latter scenario, an extra conversion step is required to transform the object
    # into mlrun.datastore.DatastoreProfile.
    if isinstance(public_profile, mlrun.common.schemas.DatastoreProfile):
        public_profile = DatastoreProfile2Json.create_from_json(
            public_json=public_profile.object
        )
    project_ds_name_private = DatastoreProfile.generate_secret_key(
        profile_name, project_name
    )
    private_body = get_secret_or_env(project_ds_name_private, secret_provider=secrets)
    if not public_profile or not private_body:
        raise mlrun.errors.MLRunNotFoundError(
            f"Unable to retrieve the datastore profile '{url}' from either the server or local environment. "
            "Make sure the profile is registered correctly, or if running in a local environment, "
            "use register_temporary_client_datastore_profile() to provide credentials locally."
        )

    datastore = DatastoreProfile2Json.create_from_json(
        public_json=DatastoreProfile2Json.get_json_public(public_profile),
        private_json=private_body,
    )
    return datastore


def register_temporary_client_datastore_profile(profile: DatastoreProfile):
    """Register the datastore profile.
    This profile is temporary and remains valid only for the duration of the caller's session.
    It's beneficial for testing purposes.
    """
    TemporaryClientDatastoreProfiles().add(profile)


def remove_temporary_client_datastore_profile(profile_name: str):
    TemporaryClientDatastoreProfiles().remove(profile_name)
