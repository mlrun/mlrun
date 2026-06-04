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

import contextlib
import threading
import time
from collections.abc import Callable, Iterator


class ThrottledSlot:
    """
    Per-process rate-limit for best-effort event publishing.

    Each independent event source instantiates its own ``ThrottledSlot`` —
    distinct event categories should not compete for the same throttle window,
    so the state intentionally lives on the instance rather than at module
    level.

    Usage::

        _slot = ThrottledSlot(
            lambda: mlrun.mlconf.events.<key>.min_emit_interval_seconds
        )

        with _slot.claim() as acquired:
            if not acquired:
                return False  # throttled — skip this emit
            client.emit(event)
        return True

    On normal exit of the ``with`` block the slot stays consumed. If the block
    raises, the slot is restored so the next attempt can retry — matching the
    intent that an *unsuccessful* delivery should not eat the throttle window.
    The interval is fetched on every claim via the caller-supplied getter so
    runtime config changes (e.g. test ``monkeypatch.setattr`` on the ``mlconf``
    field) are picked up without restarting the process.
    """

    def __init__(self, min_interval_seconds_getter: Callable[[], float]):
        """
        :param min_interval_seconds_getter: callable returning the current
            minimum interval (in seconds) between successful emissions.
        """
        self._lock = threading.Lock()
        self._last_emit_monotonic: float = 0.0
        self._get_min_interval = min_interval_seconds_getter

    @contextlib.contextmanager
    def claim(self) -> Iterator[bool]:
        """
        Context manager that yields ``True`` when the slot was claimed for this
        window and ``False`` when the call is inside the throttle window.

        On normal exit the slot stays consumed (so subsequent calls within the
        window see ``False``). If the ``with`` block raises a regular
        ``Exception`` after a successful claim, the slot is restored so the
        next attempt can retry. ``BaseException`` subclasses (``KeyboardInterrupt``,
        ``SystemExit``) skip the restore — they signal process teardown, not a
        recoverable delivery failure.
        """
        previous = self._try_claim()
        acquired = previous is not None
        try:
            yield acquired
        except Exception:
            if acquired:
                self._release(previous)
            raise

    def _try_claim(self) -> float | None:
        """Atomic check-and-set; returns the previous timestamp on success."""
        min_interval = float(self._get_min_interval())
        now = time.monotonic()
        with self._lock:
            if now - self._last_emit_monotonic < min_interval:
                return None
            previous = self._last_emit_monotonic
            self._last_emit_monotonic = now
            return previous

    def _release(self, previous: float) -> None:
        """Restore the previous timestamp; called from :meth:`claim` on failure."""
        with self._lock:
            self._last_emit_monotonic = previous
