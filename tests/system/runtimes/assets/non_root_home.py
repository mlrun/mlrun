# Copyright 2026 Iguazio
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
import json
import os
import uuid


def probe(context):
    context.log_results(_probe_cache_dir())


def handler(context, event):
    """Probe on invocation, not at init, so failures appear in the response."""
    try:
        result = _probe_cache_dir()
    except Exception as exc:
        return context.Response(
            body=json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
            content_type="application/json",
            status_code=500,
        )
    return context.Response(
        body=json.dumps(result),
        content_type="application/json",
        status_code=200,
    )


def _probe_cache_dir() -> dict:
    # write first so an unwritable HOME raises the original PermissionError
    cache_dir = os.path.expanduser("~/.cache")
    os.makedirs(cache_dir, exist_ok=True)
    probe_path = os.path.join(cache_dir, f"mlrun-non-root-home-{uuid.uuid4().hex}")
    with open(probe_path, "w") as probe_file:
        probe_file.write("ok")
    os.remove(probe_path)
    return {
        "uid": os.getuid(),
        "home": os.environ.get("HOME"),
        "cache_dir": cache_dir,
        "writable": True,
    }
