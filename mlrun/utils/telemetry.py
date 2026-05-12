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

import os

import mlrun.common.constants
import mlrun.errors
import mlrun.utils


def resolve_otlp_headers(path: str | None = None) -> dict[str, str]:
    """Read OTLP auth headers from a mounted directory.

    Each file in the directory becomes one HTTP header — filename = header name,
    file contents = header value. Hidden entries (``..data`` and friends, which
    kubelet uses for atomic secret updates) and subdirectories are skipped.

    Returns an empty dict when the directory does not exist or cannot be read —
    callers should treat the result as "no auth headers" and proceed.

    :param path: Directory to read headers from. Defaults to
                 ``mlrun.common.constants.MLRUN_TELEMETRY_OTLP_HEADERS_PATH``.
    :returns: Mapping of header name → header value, or ``{}``.
    """
    headers_path = path or mlrun.common.constants.MLRUN_TELEMETRY_OTLP_HEADERS_PATH
    if not os.path.isdir(headers_path):
        return {}

    headers: dict[str, str] = {}
    try:
        for entry in os.scandir(headers_path):
            if entry.name.startswith("."):
                continue
            if not entry.is_file():
                continue
            with open(entry.path) as f:
                headers[entry.name] = f.read().rstrip("\n")
    except OSError as exc:
        mlrun.utils.logger.warning(
            "Failed to read OTLP telemetry headers from mount",
            path=headers_path,
            error=mlrun.errors.err_to_str(exc),
        )
        return {}

    if headers:
        mlrun.utils.logger.debug(
            "Resolved OTLP telemetry headers from mount",
            path=headers_path,
            header_keys=sorted(headers.keys()),
        )
    return headers
