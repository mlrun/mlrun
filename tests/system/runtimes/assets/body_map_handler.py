# Copyright 2026 Iguazio
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


def process_mapped_data(
    body, user_name: str, user_email: str, book_titles: list
) -> dict:
    """Handler that receives mapped parameters from body_map."""
    return {
        "name": user_name,
        "email": user_email,
        "titles": book_titles,
        "count": len(book_titles),
    }


def echo_kwargs(body, **kwargs) -> dict:
    """Return all extracted kwargs — used to verify body map merging."""
    return kwargs
