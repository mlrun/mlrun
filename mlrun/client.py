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

import mlrun.errors

# No top-level runtime ``import mlrun.*`` — would race with
# ``mlrun.config._populate()``'s deferred ``from mlrun.db import get_run_db``.
# ``mlrun.errors`` is a leaf module (stdlib + aiohttp + requests only) and safe.


@dataclass(frozen=True)
class Credentials:
    """User credentials for MLRun API access.

    One of: ``token=``, ``username=/password=``, or ``use_env=True``
    for legacy env/config/file resolution.
    """

    token: str | None = None
    username: str | None = None
    password: str | None = None
    use_env: bool = False

    def __post_init__(self):
        if (self.username is None) != (self.password is None):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Basic auth requires both username and password."
            )
        active = [
            name
            for name, on in (
                ("token", self.token is not None),
                (
                    "basic_auth",
                    self.username is not None or self.password is not None,
                ),
                ("env", self.use_env),
            )
            if on
        ]
        if not active:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Credentials need an auth mode. Use Credentials(token=...), "
                "Credentials(username=..., password=...), "
                "or Credentials(use_env=True)."
            )
        if len(active) > 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Credentials require exactly one auth mode; got: {active}."
            )


_active_client: ContextVar[Client | None] = ContextVar("_active_client", default=None)


def get_active_client() -> Client | None:
    """Return the active ``Client`` for this task/thread, or ``None``."""
    return _active_client.get()


class Client:
    """A per-session MLRun client owning its own ``HTTPRunDB``.

    The backend URL is taken from ``mlrun.mlconf.dbpath`` (already populated
    by ``import mlrun``); a single MLRun cluster per Python process is
    assumed. Only credentials vary per ``Client``.

    Example::

        client = mlrun.Client(credentials=mlrun.Credentials(token="..."))
        with client.session():
            project = mlrun.get_or_create_project("my-proj")
    """

    def __init__(self, credentials: Credentials):
        # Deferred imports — see module-level comment.
        from mlrun.config import config
        from mlrun.db.httpdb import HTTPRunDB

        self._http_db = HTTPRunDB(config.dbpath, credentials=credentials)

    @contextmanager
    def session(self) -> Iterator[Client]:
        """Bind this client as active for the current contextvars scope."""
        token = _active_client.set(self)
        try:
            yield self
        finally:
            _active_client.reset(token)
