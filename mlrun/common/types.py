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

import enum

# Alias to Python's built-in StrEnum (Python 3.11+)
StrEnum = enum.StrEnum


class HTTPMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"
    PATCH = "PATCH"
    PUT = "PUT"


class Operation(StrEnum):
    ADD = "add"
    REMOVE = "remove"


class AuthenticationMode(StrEnum):
    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    IGUAZIO = "iguazio"
    IGUAZIO_V4 = "iguazio-v4"
