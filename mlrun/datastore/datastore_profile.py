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
import ast
import base64
import json
import typing
from urllib.parse import ParseResult, urlparse, urlunparse

import pydantic
from mergedeep import merge

import mlrun
import mlrun.errors

from ..secrets import get_secret_or_env


class DatastoreProfile(pydantic.BaseModel):
    type: str
    name: str
    _private_attributes: typing.List = ()

    @pydantic.validator("name")
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


class TemporaryClientDatastoreProfiles(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._data = {}  # Initialize the dictionary

    def add(self, profile: DatastoreProfile):
        self._data[profile.name] = profile

    def get(self, key):
        return self._data.get(key, None)


class DatastoreProfileBasic(DatastoreProfile):
    type: str = pydantic.Field("basic")
    _private_attributes = "private"
    public: str
    private: typing.Optional[str] = None


class DatastoreProfileKafkaTarget(DatastoreProfile):
    type: str = pydantic.Field("kafka_target")
    _private_attributes = "kwargs_private"
    bootstrap_servers: str
    topic: str
    kwargs_public: typing.Optional[typing.Dict]
    kwargs_private: typing.Optional[typing.Dict]

    def attributes(self):
        attributes = {"bootstrap_servers": self.bootstrap_servers}
        if self.kwargs_public:
            attributes = merge(attributes, self.kwargs_public)
        if self.kwargs_private:
            attributes = merge(attributes, self.kwargs_private)
        return attributes


class DatastoreProfileKafkaSource(DatastoreProfile):
    type: str = pydantic.Field("kafka_source")
    _private_attributes = ("kwargs_private", "sasl_user", "sasl_pass")
    brokers: typing.Union[str, typing.List[str]]
    topics: typing.Union[str, typing.List[str]]
    group: typing.Optional[str] = "serving"
    initial_offset: typing.Optional[str] = "earliest"
    partitions: typing.Optional[typing.Union[str, typing.List[str]]]
    sasl_user: typing.Optional[str]
    sasl_pass: typing.Optional[str]
    kwargs_public: typing.Optional[typing.Dict]
    kwargs_private: typing.Optional[typing.Dict]

    def attributes(self):
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
        sasl = attributes.pop("sasl", {})
        if self.sasl_user and self.sasl_pass:
            sasl["enabled"] = True
            sasl["user"] = self.sasl_user
            sasl["password"] = self.sasl_pass
        if sasl:
            attributes["sasl"] = sasl
        return attributes


class DatastoreProfileS3(DatastoreProfile):
    type: str = pydantic.Field("s3")
    _private_attributes = ("access_key", "secret_key")
    endpoint_url: typing.Optional[str] = None
    force_non_anonymous: typing.Optional[str] = None
    profile_name: typing.Optional[str] = None
    assume_role_arn: typing.Optional[str] = None
    access_key: typing.Optional[str] = None
    secret_key: typing.Optional[str] = None


class DatastoreProfileRedis(DatastoreProfile):
    type: str = pydantic.Field("redis")
    _private_attributes = ("username", "password")
    endpoint_url: str
    username: typing.Optional[str] = None
    password: typing.Optional[str] = None

    def is_secured(self):
        return self.endpoint_url.startswith("rediss://")

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
        return urlunparse(new_parsed_url)


class DatastoreProfile2Json(pydantic.BaseModel):
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
                if not str(k) in profile._private_attributes
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
        ds_profile_factory = {
            "s3": DatastoreProfileS3,
            "redis": DatastoreProfileRedis,
            "basic": DatastoreProfileBasic,
            "kafka_target": DatastoreProfileKafkaTarget,
            "kafka_source": DatastoreProfileKafkaSource,
        }
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


def datastore_profile_read(url):
    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() != "ds":
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"resource URL '{url}' cannot be read as a datastore profile because its scheme is not 'ds'"
        )

    profile_name = parsed_url.hostname
    project_name = parsed_url.username or mlrun.mlconf.default_project
    datastore = TemporaryClientDatastoreProfiles().get(profile_name)
    if datastore:
        return datastore
    public_profile = mlrun.db.get_run_db().get_datastore_profile(
        profile_name, project_name
    )
    project_ds_name_private = DatastoreProfile.generate_secret_key(
        profile_name, project_name
    )
    private_body = get_secret_or_env(project_ds_name_private)
    if not public_profile or not private_body:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Unable to retrieve the datastore profile '{url}' from either the server or local environment."
            "Make sure the profile is registered correctly, or if running in a local environment,"
            "use register_temporary_client_datastore_profile() to provide credentials locally."
        )

    datastore = DatastoreProfile2Json.create_from_json(
        public_json=DatastoreProfile2Json.get_json_public(public_profile),
        private_json=private_body,
    )
    return datastore


def register_temporary_client_datastore_profile(profile: DatastoreProfile):
    TemporaryClientDatastoreProfiles().add(profile)


def datastore_profile_embed_url_scheme(url):
    profile = datastore_profile_read(url)
    parsed_url = urlparse(url)
    scheme = profile.type
    # Add scheme as a password to the network location part
    netloc = f"{parsed_url.username or ''}:{scheme}@{parsed_url.netloc}"

    # Construct the new URL
    new_url = urlunparse(
        [
            parsed_url.scheme,
            netloc,
            parsed_url.path,
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment,
        ]
    )
    return new_url
