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

from mlrun.utils import logger

import framework.utils.auth.providers.opa as opa
import services.api.utils.events.events_factory as events_factory


def publish_permission_denied(
    resource: str,
    action: str,
    username: str | None = None,
) -> bool:
    """
    Best-effort publish of a ``UserAction.General.PermissionDenied`` event.

    :return: True if an event was emitted, False if unsupported (e.g. a no-op
        client in a v3 environment or events disabled) or delivery failed.
    """
    try:
        client = events_factory.EventsFactory.get_events_client()
        event = client.generate_permission_denied_event(
            resource=resource, action=action, username=username
        )
        if event is None:
            return False
        client.emit(event)
        return True
    except Exception as publish_exc:
        logger.warning(
            "Failed to publish permission denied event",
            resource=resource,
            action=action,
            exc_info=publish_exc,
        )
        return False


def register_for_opa_provider() -> None:
    """
    Attach ``publish_permission_denied`` as the OPA provider's permission-denied
    listener. Safe to call multiple times.
    """
    opa.register_permission_denied_listener(publish_permission_denied)
