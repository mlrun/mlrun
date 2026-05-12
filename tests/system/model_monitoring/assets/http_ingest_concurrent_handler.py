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

Fires N prediction events **simultaneously** to the stream pod using a
ThreadPoolExecutor, times the batch wall-clock duration, and returns it.

The stream pod uses storey ConcurrentExecution, so N concurrent in-cluster
HTTP requests should complete in roughly the same time as one.  If the pod
serialised requests the elapsed time would grow linearly with N.

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
    num_events = int(body.get("num_events", 20))

    monitoring_url = _MONITORING_URL.rstrip("/")

    payloads = [
        {
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
        for i in range(num_events)
    ]

    def _post(payload: dict) -> int:
        return requests.post(monitoring_url, json=payload, timeout=15).status_code

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=num_events) as executor:
        statuses = list(executor.map(_post, payloads))
    elapsed = time.monotonic() - start

    pushed = sum(1 for s in statuses if s == 202)
    context.logger.info(
        f"Concurrent push: {pushed}/{num_events} accepted in {elapsed:.2f}s"
    )
    return context.Response(
        body=json.dumps({"pushed": pushed, "elapsed_seconds": elapsed}),
        status_code=200,
        content_type="application/json",
    )