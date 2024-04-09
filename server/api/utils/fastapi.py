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

import mlrun.errors


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
        try:
            parameters = {}

            # if a query parameter is a list, the dict will flatten it to a single value
            # so we need to access the query params from the _list attribute
            for param, value in request.query_params._list:
                parsed_param = param.replace("-", "_")
                if parsed_param in model.__fields__:

                    # check the param type in the schema and convert the value to the correct type
                    param_type = model.__fields__[parsed_param].type_
                    if param_type == bool:
                        value = value.lower() in ["true", "1", "yes", "t", "y"]
                    elif typing.get_origin(param_type) != typing.Union:
                        value = model.__fields__[parsed_param].type_(value)

                # add the value to the parameters dict while handling multiple values
                if param in parameters:
                    if not isinstance(parameters[param], list):
                        parameters[param] = [parameters[param]]
                    parameters[param].append(value)
                else:
                    parameters[param] = value

            return model(**parameters)
        except (pydantic.ValidationError, ValueError) as e:
            raise mlrun.errors.MLRunUnprocessableEntityError from e

    return wrapper
