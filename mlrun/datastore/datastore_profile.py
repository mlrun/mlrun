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

import mlrun
import mlrun.errors

from ..secrets import get_secret_or_env


class PrivateValue(pydantic.BaseModel):
    value: str

    def get(self):
        if self.value == "None":
            return None
        return ast.literal_eval(self.value)["value"]


class DatastoreProfile(pydantic.BaseModel):
    type: str
    name: str

    @pydantic.validator("name")
    def lower_case(cls, v):
        return v.lower()

    @staticmethod
    def generate_secret_key(profile_name: str, project: str):
        secret_name_separator = "-__-"
        full_key = (
            "datastore-profiles"
            + secret_name_separator
            + project
            + secret_name_separator
            + profile_name
        )
        return full_key


class DatastoreProfileRedis(DatastoreProfile):
    type: str = pydantic.Field("redis")
    endpoint_url: str
    username: typing.Optional[PrivateValue]
    password: typing.Optional[PrivateValue]

    @pydantic.validator("username", "password", pre=True)
    def convert_to_private(cls, v):
        return PrivateValue(value=v)

    def is_secured(self):
        return self.endpoint_url.startswith("rediss://")

    def url_with_credentials(self):
        parsed_url = urlparse(self.endpoint_url)
        username = self.username.get() if self.username else None
        password = self.password.get() if self.password else None
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
            {k: v for k, v in profile.dict().items() if not isinstance(v, dict)}
        )

    @staticmethod
    def get_json_private(profile: DatastoreProfile) -> str:
        return DatastoreProfile2Json._to_json(
            {k: v for k, v in profile.dict().items() if isinstance(v, dict)}
        )

    @staticmethod
    def create_from_json(public_json: str, private_json: str = "{}"):
        attr1 = json.loads(public_json)
        attr2 = json.loads(private_json)
        attributes = {**attr1, **attr2}
        decoded_dict = {
            k: base64.b64decode(str(v).encode()).decode() for k, v in attributes.items()
        }
        datastore_type = decoded_dict.get("type")
        if datastore_type == "redis":
            return DatastoreProfileRedis.parse_obj(decoded_dict)
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
    public_profile = mlrun.db.get_run_db().get_datastore_profile(
        profile_name, project_name
    )
    project_ds_name_private = DatastoreProfile.generate_secret_key(
        profile_name, project_name
    )
    private_body = get_secret_or_env(project_ds_name_private)

    datastore = DatastoreProfile2Json.create_from_json(
        public_json=DatastoreProfile2Json.get_json_public(public_profile),
        private_json=private_body,
    )
    return datastore
