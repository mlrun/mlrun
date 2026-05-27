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

"""Per-session MLRun client for multi-user / multi-session usage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

# No top-level runtime ``import mlrun.*`` — would race with
# ``mlrun.config._populate()``'s deferred ``from mlrun.db import get_run_db``.
if TYPE_CHECKING:
    import mlrun.auth


@dataclass(frozen=True)
class Credentials:
    """User credentials for MLRun API access.

    One of: ``token=``, ``token_provider=``, ``username=/password=``, or
    ``Credentials.from_env()`` for legacy env/config/file resolution.
    """

    token: str | None = None
    token_provider: mlrun.auth.TokenProvider | None = None
    username: str | None = None
    password: str | None = None
    _use_env: bool = False

    @classmethod
    def from_env(cls) -> Credentials:
        """Delegate auth resolution to ``HTTPRunDB``'s legacy env/config/file path."""
        return cls(_use_env=True)


_active_client: ContextVar[Client | None] = ContextVar("_active_client", default=None)


def get_active_client() -> Client | None:
    """Return the active ``Client`` for this task/thread, or ``None``."""
    return _active_client.get()


class Client:
    """A per-session MLRun client owning its own ``HTTPRunDB``.

    Example::

        client = mlrun.Client(dbpath="...", credentials=mlrun.Credentials(token="..."))
        with client.session():
            project = mlrun.get_or_create_project("my-proj")
    """

    def __init__(self, dbpath: str, credentials: Credentials):
        import mlrun.db.httpdb  # deferred; see module-level comment

        self._http_db = mlrun.db.httpdb.HTTPRunDB(dbpath, credentials=credentials)
        self._http_db.connect()

    @contextmanager
    def session(self) -> Iterator[Client]:
        """Bind this client as active for the current contextvars scope."""
        token = _active_client.set(self)
        try:
            yield self
        finally:
            _active_client.reset(token)
