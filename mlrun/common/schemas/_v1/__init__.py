# Copyright 2024 Iguazio
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
"""Pydantic 1 schema face.

Today's schema models, relocated here unchanged (defined on the ``pydantic.v1``
namespace) — what the client/SDK loads, and what the per-topic facades bind under
Pydantic 1. Imports version-agnostic definitions from ``_shared`` and peer models
from sibling ``_v1.<topic>`` modules; this package has no re-exports of its own.
"""
