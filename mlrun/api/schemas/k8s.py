# Copyright 2018 Iguazio
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

import mlrun.api.utils.helpers


class ResourceSpec(pydantic.BaseModel):
    cpu: typing.Optional[str]
    memory: typing.Optional[str]
    gpu: typing.Optional[str]


class Resources(pydantic.BaseModel):
    requests: ResourceSpec = ResourceSpec()
    limits: ResourceSpec = ResourceSpec()


class NodeSelectorOperator(mlrun.api.utils.helpers.StrEnum):
    """
    A node selector operator is the set of operators that can be used in a node selector requirement
    https://github.com/kubernetes/api/blob/b754a94214be15ffc8d648f9fe6481857f1fc2fe/core/v1/types.go#L2765
    """

    node_selector_op_in = "In"
    node_selector_op_not_in = "NotIn"
    node_selector_op_exists = "Exists"
    node_selector_op_does_not_exist = "DoesNotExist"
    node_selector_op_gt = "Gt"
    node_selector_op_lt = "Lt"
