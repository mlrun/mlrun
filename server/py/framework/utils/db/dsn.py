# Copyright 2025 Iguazio
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
import re
from typing import Any, Optional, Union
from urllib.parse import parse_qs, quote, unquote, urlparse

import sqlalchemy

import mlrun
import mlrun.common.db.dialects
import mlrun.utils.helpers


class Dsn:
    _IDENTIFIER_REGEX = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")  # driver
    _HOST_REGEX = re.compile(r"(?:[A-Za-z0-9.\-]+|\[[0-9A-Fa-f:]+\])")  # host / IPv6
    # SQLite path may not start with “/” and must not contain “..”
    _PATH_REGEX = re.compile(r"(?!/)(?!.*\.\.)[A-Za-z0-9_\-./]+")
    _DBNAME_REGEX = re.compile(r"[A-Za-z0-9_\-\.$]+")  # db‑name
    _SAFE_USERINFO_CHARS = "~/"  # characters we do NOT percent‑encode in user‑info

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        escaped_dsn = self._encode_credentials(dsn)
        self._parsed = urlparse(escaped_dsn)
        self.dialect, self.driver = self._split_scheme(self._parsed.scheme)

        # Connection components
        if self.dialect == mlrun.common.db.dialects.Dialects.SQLITE:
            # For SQLite the path is part of the file name, not the DB name
            self.username = self.password = self.host = self.port = self.database = None
        else:
            self.username = (
                unquote(self._parsed.username) if self._parsed.username else None
            )
            if self._parsed.password:
                # first pass undoes our own %25 escaping, second pass undoes user‑supplied escapes
                pwd_once_unquoted = unquote(self._parsed.password)
                self.password = unquote(pwd_once_unquoted)
            else:
                self.password = None
            self.host = self._parsed.hostname
            try:
                # urlparse.port raises ValueError for non‑numeric ports
                self.port = self._parsed.port
            except ValueError:
                self.port = None
            self.database = self._parsed.path.lstrip("/") or None

        # Query configurations
        if self._parsed.query:
            raw_qs = parse_qs(self._parsed.query)
            self.configurations: dict[str, Union[str, list[str]]] = {
                key: (value[0] if len(value) == 1 else value)
                for key, value in raw_qs.items()
            }
        else:
            self.configurations = {}

    def _encode_credentials(self, dsn: str) -> str:
        """
        Percent‑encode the user‑info (everything up to, but not including, the
        final “@” in the authority component) so that urlparse() will not split
        on reserved characters such as '#', '?', '|', '<', etc.
        """
        if "://" not in dsn or "@" not in dsn:
            return dsn
        scheme, url_after_scheme = dsn.split("://", 1)
        # split on the *last* “@” to tolerate raw “@” inside the password
        userinfo, _, host_port = url_after_scheme.rpartition("@")
        if not userinfo:  # no credentials section
            return dsn
        if ":" in userinfo:
            user, pwd = userinfo.split(":", 1)
            encoded_userinfo = (
                f"{quote(user, safe=self._SAFE_USERINFO_CHARS)}:"
                f"{quote(pwd, safe=self._SAFE_USERINFO_CHARS)}"
            )
        else:
            encoded_userinfo = quote(userinfo, safe=self._SAFE_USERINFO_CHARS)
        return f"{scheme}://{encoded_userinfo}@{host_port}"

    def is_valid(self) -> bool:
        """
        DSN validator

        1. Dialect must be known.
        2. Driver (if present) must be a valid identifier.
        3. SQLite DSNs: validate the optional path and return early.
        4. Other DBs: validate database name, user/host/port.
        """
        if self.dialect not in mlrun.common.db.dialects.Dialects.all():
            return False

        if self.driver and not self._IDENTIFIER_REGEX.fullmatch(self.driver):
            return False

        # The authority part must contain exactly *one* raw “@”
        url_after_scheme = self._dsn.split("://", 1)[-1]
        if url_after_scheme.count("@") != 1:
            return False

        if self.dialect == mlrun.common.db.dialects.Dialects.SQLITE:
            raw_path = self._parsed.path.lstrip("/")
            return (
                not raw_path
                or raw_path == ":memory:"
                or self._PATH_REGEX.fullmatch(raw_path)
            )

        if self.database is None or not self._DBNAME_REGEX.fullmatch(self.database):
            return False
        if not self.username:  # username required
            return False
        if not self.host or not self._HOST_REGEX.fullmatch(self.host):
            return False
        if not mlrun.utils.helpers.is_valid_port(self.port):
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "dialect": self.dialect,
            "driver": self.driver,
            "username": self.username,
            "password": self.password,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "configurations": self.configurations,
        }

    def as_sqlalchemy_dsn(self):
        drivername = (
            self.dialect if self.driver is None else f"{self.dialect}+{self.driver}"
        )
        return sqlalchemy.URL(
            drivername=drivername,
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
            query=self.configurations or None,  # keep query params
        )

    def __str__(self) -> str:
        return self.as_sqlalchemy_dsn().render_as_string()

    @staticmethod
    def _split_scheme(scheme: str) -> tuple[str, Optional[str]]:
        parts = scheme.split("+", 1)
        return parts[0], parts[1] if len(parts) == 2 else None
