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


def parse_versioned_object_uri(
    uri: str, default_project: str = ""
) -> tuple[str, str, str, str]:
    project = default_project
    tag = ""
    hash_key = ""
    if "/" in uri:
        loc = uri.find("/")
        project = uri[:loc]
        uri = uri[loc + 1 :]
    if ":" in uri:
        loc = uri.find(":")
        tag = uri[loc + 1 :]
        uri = uri[:loc]
    if "@" in uri:
        loc = uri.find("@")
        hash_key = uri[loc + 1 :]
        uri = uri[:loc]

    return project, uri, tag, hash_key


def generate_api_gateway_name(project: str, name: str) -> str:
    """
    Generate a unique (within project) api gateway name
    :param project: project name
    :param name: api gateway name

    :return: the resolved api gateway name
    """
    return f"{project}-{name}" if project else name


def dummy_function(x, y, operation="add"):
    """
    A dummy function that performs basic operations on x and y.
    """
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        print("Both x and y must be numbers.")
        return None

    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    elif operation == "multiply":
        return x * y
    elif operation == "divide":
        if y == 0:
            print("Cannot divide by zero.")
            return None
        return x / y
    else:
        print(f"Unsupported operation: {operation}")
        return None
