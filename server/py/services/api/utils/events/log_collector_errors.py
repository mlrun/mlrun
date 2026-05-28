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

import asyncio

import mlrun
import mlrun.common.schemas
from mlrun.utils import logger

import framework.utils.clients.log_collector as log_collector_client
import services.api.utils.events.events_factory as events_factory
import services.api.utils.events.throttle as throttle

_slot = throttle.ThrottledSlot(
    lambda: mlrun.mlconf.events.log_collector.min_emit_interval_seconds
)


def publish_log_collector_failed(
    operation: str | None = None,
    run_uid: str | None = None,
    project: str | None = None,
    error: BaseException | str | None = None,
    error_code: int | str | None = None,
    error_category: str | None = None,
) -> bool:
    """
    Best-effort publish of a ``MLRun.LogCollector.Failed`` event.

    Throttled to one emission per process per
    ``mlconf.events.log_collector.min_emit_interval_seconds``. The throttle slot
    is consumed only on successful delivery; a no-op client (e.g. v3 environment
    or events disabled) or a raising ``emit`` leave the slot free so the next
    log-collector failure can retry.

    :return: True if an event was emitted, False if throttled or unsupported.
    """
    try:
        client = events_factory.EventsFactory.get_events_client()
        event = client.generate_log_collector_event(
            action=mlrun.common.schemas.LogCollectorEventActions.failed,
            error=error,
            error_category=error_category,
            error_code=error_code,
            operation=operation,
            run_uid=run_uid,
            project=project,
        )
        if event is None:
            return False
        with _slot.claim() as acquired:
            if not acquired:
                return False
            client.emit(event)
        return True
    except Exception as publish_exc:
        logger.warning(
            "Failed to publish log collector failed event",
            operation=operation,
            run_uid=run_uid,
            project=project,
            error_category=error_category,
            error_code=error_code,
            exc_info=publish_exc,
        )
        return False


def register_for_log_collector() -> None:
    """
    Attach the log-collector failure listener to the framework client. Safe to
    call multiple times — the framework helper deduplicates by identity.
    """
    log_collector_client.add_failure_listener(_on_log_collector_failure)


def _on_log_collector_failure(context: dict) -> None:
    """
    Failure listener registered on the framework log-collector client.

    The framework client is exercised from ``async def`` retrieval RPCs, so this
    listener offloads the (synchronous) HTTP emit to the default executor to
    avoid blocking the event loop. Falls back to an inline publish when there is
    no running loop (e.g. tests invoking the listener directly).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        publish_log_collector_failed(
            operation=context.get("operation"),
            run_uid=context.get("run_uid"),
            project=context.get("project"),
            error=context.get("error"),
            error_code=context.get("error_code"),
            error_category=context.get("error_category"),
        )
        return
    loop.run_in_executor(
        None,
        lambda: publish_log_collector_failed(
            operation=context.get("operation"),
            run_uid=context.get("run_uid"),
            project=context.get("project"),
            error=context.get("error"),
            error_code=context.get("error_code"),
            error_category=context.get("error_category"),
        ),
    )
