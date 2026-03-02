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


def process_path_and_query_params(
    category: str, item_id: str, tags: list | None = None, limit: str | None = None
) -> dict:
    """Handler that receives path parameters and query parameters.

    Args:
        category: From path parameter {category}
        item_id: From path parameter {item_id}
        tags: From repeated query param ?tags=...&tags=... (list)
        limit: From single query param ?limit=... (string)

    Returns:
        Dict with all parameters for verification
    """
    return {
        "category": category,
        "item_id": item_id,
        "tags": tags or [],
        "limit": limit,
        "tags_count": len(tags) if tags else 0,
    }
