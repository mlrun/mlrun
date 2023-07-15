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


class privateValue(pydantic.BaseModel):
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


class DatastoreProfileRedis(DatastoreProfile):
    type: str = pydantic.Field("redis")
    endpoint_url: str
    username: typing.Optional[privateValue]
    password: typing.Optional[privateValue]

    @pydantic.validator("username", "password", pre=True)
    def convert_to_private(cls, v):
        return privateValue(value=v)

    def is_secured(self):
        return self.endpoint_url.startswith("rediss://")

    def url_with_credentials(self):
        parsed_url = urlparse(self.endpoint_url)
        if self.username.get() and self.password.get():
            netloc = (
                f"{self.username.get()}:{self.password.get()}@{parsed_url.hostname}"
            )
        elif self.username.get():
            netloc = f"{self.username.get()}@{parsed_url.hostname}"
        else:
            netloc = parsed_url.hostname

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
        if decoded_dict["type"] == "redis":
            return DatastoreProfileRedis.parse_obj(decoded_dict)
        return None


def datastore_profile_read(url):
    parsed_url = urlparse(url)
    if parsed_url.scheme.lower() != "ds":
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"resource {url} does not have datastore profile format"
        )

    profile_name = parsed_url.hostname
    project_name = parsed_url.username or mlrun.mlconf.default_project
    profile = mlrun.db.get_run_db().get_datastore_profile(profile_name, project_name)
    public_wrapper = json.loads(profile._content)

    project_ds_name_private = mlrun.api.crud.DatastoreProfiles.generate_secret_key(
        profile_name, project_name
    )
    private_body = get_secret_or_env(project_ds_name_private)

    datastore = DatastoreProfile2Json.create_from_json(
        public_json=public_wrapper["body"], private_json=private_body
    )
    return datastore
