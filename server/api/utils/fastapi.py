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
from fastapi import Request
from starlette.datastructures import MultiDict


def convert_query_params_to_snake_case(request: Request) -> None:
    query_params = request.query_params
    new_params = MultiDict()
    for k, v in query_params.multi_items():
        if "-" in k:
            snake = "_".join(k.split("-"))
            new_params.append(snake, v)
        else:
            new_params.append(k, v)
    request._query_params = new_params
