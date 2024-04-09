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
import typing

import pydantic
from fastapi import Request


class SchemaModifiers:
    defaults = {"enforce_kebab_case": True}

    @classmethod
    def get_schema_annotations(cls, **annotations) -> dict:
        schema_annotations = cls.defaults.copy()
        schema_annotations.update(annotations)
        return {"annotations": schema_annotations}

    @staticmethod
    def enforce_kebab_case(method_data: dict) -> dict:
        parameters = method_data.get("parameters", [])
        for parameter in parameters:
            parameter["name"] = parameter["name"].replace("_", "-")
        return method_data


def schema_as_query_parameter_definition(
    model: pydantic.main.ModelMetaclass,
) -> typing.Callable:
    def wrapper(request: Request) -> pydantic.BaseModel:
        parameters = {}
        for param, value in request.query_params._list:
            if param in parameters:
                if not isinstance(parameters[param], list):
                    parameters[param] = [parameters[param]]
                parameters[param].append(value)
            else:
                parameters[param] = value
        return model(**parameters)

    return wrapper
