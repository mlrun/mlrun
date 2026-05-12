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

"""Nuclio handler used by test_http_ingest_stream_pod_is_async.

Supports two modes controlled by the request body:
    {"single": true}       — send one event, return elapsed as single_elapsed
    {"num_events": N}      — send N events concurrently, return elapsed as concurrent_elapsed

The test runs single first to get a per-request baseline, then concurrent.
Asserting concurrent_elapsed < single_elapsed * 2 proves the stream pod
handles requests in parallel rather than serialising them.

Env vars (injected by MLRun at deploy time):
    MODEL_MONITORING_URL  — HTTP URL of the monitoring stream pod.
    MODEL_ENDPOINT_UID    — UID of the primary model endpoint.
    MODEL_ENDPOINT_NAME   — Name of the primary model endpoint.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

_MONITORING_URL = os.environ.get("MODEL_MONITORING_URL", "")
_ENDPOINT_UID = os.environ.get("MODEL_ENDPOINT_UID", "")
_ENDPOINT_NAME = os.environ.get("MODEL_ENDPOINT_NAME", "")


def _build_payload(i: int) -> dict:
    return {
        "model_endpoint_uid": _ENDPOINT_UID,
        "model_endpoint_name": _ENDPOINT_NAME,
        "inputs": {
            "age": float(i),
            "income": float(i + 1),
            "credit_score": float(i + 2),
            "balance": float(i + 3),
        },
        "outputs": {"approved": float(i % 2)},
    }


def _post(payload: dict) -> int:
    return requests.post(
        _MONITORING_URL.rstrip("/"), json=payload, timeout=15
    ).status_code


def handler(context, event):
    if not _MONITORING_URL:
        return context.Response(
            body="MODEL_MONITORING_URL env var is not set",
            status_code=503,
            content_type="text/plain",
        )
    if not _ENDPOINT_UID:
        return context.Response(
            body="MODEL_ENDPOINT_UID env var is not set",
            status_code=503,
            content_type="text/plain",
        )

    body = event.body
    if isinstance(body, (bytes, bytearray)):
        body = json.loads(body) if body else {}
    body = body or {}

    if body.get("single"):
        start = time.monotonic()
        status = _post(_build_payload(0))
        elapsed = time.monotonic() - start
        context.logger.info(f"Single request: status={status} elapsed={elapsed:.3f}s")
        return context.Response(
            body=json.dumps(
                {"pushed": 1 if status == 202 else 0, "single_elapsed": elapsed}
            ),
            status_code=200,
            content_type="application/json",
        )

    num_events = int(body.get("num_events", 20))
    payloads = [_build_payload(i) for i in range(num_events)]

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=num_events) as executor:
        statuses = list(executor.map(_post, payloads))
    elapsed = time.monotonic() - start

    pushed = sum(1 for s in statuses if s == 202)
    context.logger.info(
        f"Concurrent push: {pushed}/{num_events} accepted in {elapsed:.3f}s"
    )
    return context.Response(
        body=json.dumps({"pushed": pushed, "concurrent_elapsed": elapsed}),
        status_code=200,
        content_type="application/json",
    )
