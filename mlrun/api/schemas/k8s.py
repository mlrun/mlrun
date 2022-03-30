import enum
import typing

import pydantic


class ResourceSpec(pydantic.BaseModel):
    cpu: typing.Optional[str]
    memory: typing.Optional[str]
    gpu: typing.Optional[str]


class Resources(pydantic.BaseModel):
    requests: ResourceSpec = ResourceSpec()
    limits: ResourceSpec = ResourceSpec()


class NodeSelectorOperator(str, enum.Enum):
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
