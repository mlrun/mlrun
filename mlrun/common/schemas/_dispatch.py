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
"""Internal helper: which schema face the package binds.

This is an **environment** decision, not an install-version one. A client can run
on Pydantic 2 yet must keep the ``_v1`` models, because its call-sites still use
the Pydantic-1 instance API (``.dict()`` / ``.json()`` / ``.parse_obj()`` / …).
Only the mlrun **API server** runs the native Pydantic-2 face (``_v2``), and only
once its call-sites are migrated. ``MLRUN_IS_API_SERVER`` is that environment
signal — the same one ``mlrun.config.is_running_as_api()`` reads, inlined here as
a plain ``os.getenv`` so importing this module never pulls ``mlrun.config`` (a
higher-level ``mlrun.*``) into ``mlrun.common`` at import time.
"""

import os

# True inside the mlrun API server process (set by the ``mlrun db`` entrypoint).
IS_API_SERVER = os.getenv("MLRUN_IS_API_SERVER", "false").lower() == "true"

# ``_v2`` is an empty placeholder until ML-12891 adds the native models, and the
# API server can only run them once its call-sites are migrated and Pydantic 2 is
# installed. ML-12892 removes this guard — binding the API server to ``_v2`` — as
# part of that migration. Until then every environment (the API server included)
# binds ``_v1``, so the empty ``_v2`` face is never selected and the split stays
# dormant.
_V2_FACE_ENABLED = False

# The single switch the package dispatchers read: bind ``_v2`` only in the API
# server environment, and only once the v2 face is enabled.
USE_V2_SCHEMAS = IS_API_SERVER and _V2_FACE_ENABLED
