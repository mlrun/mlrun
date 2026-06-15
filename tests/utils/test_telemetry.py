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

from unittest.mock import MagicMock

import mlrun.common.constants
import mlrun.utils.telemetry


def test_returns_empty_when_mount_path_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(missing)) == {}


def test_returns_empty_when_mount_path_is_empty_dir(tmp_path):
    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(tmp_path)) == {}


def test_reads_single_header(tmp_path):
    (tmp_path / "Authorization").write_text("Bearer token-xyz")

    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(tmp_path)) == {
        "Authorization": "Bearer token-xyz"
    }


def test_reads_multiple_headers(tmp_path):
    (tmp_path / "Authorization").write_text("Bearer token-xyz")
    (tmp_path / "X-Scope-OrgID").write_text("tenant-42")

    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(tmp_path)) == {
        "Authorization": "Bearer token-xyz",
        "X-Scope-OrgID": "tenant-42",
    }


def test_strips_trailing_newline_from_header_values(tmp_path):
    """`kubectl create secret --from-file=...` writes the file contents verbatim,
    which usually ends in '\\n'. HTTP forbids newlines in header values, so the
    helper must strip a trailing newline."""
    (tmp_path / "Authorization").write_text("Bearer token-xyz\n")
    (tmp_path / "X-Multi-Newline").write_text("value\n\n")
    (tmp_path / "X-No-Newline").write_text("value-without-newline")

    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(tmp_path)) == {
        "Authorization": "Bearer token-xyz",
        "X-Multi-Newline": "value",
        "X-No-Newline": "value-without-newline",
    }


def test_decodes_complex_header_values(tmp_path):
    """Header values with spaces, JWT-style payloads, non-ASCII bytes, and empty values."""
    payload = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig",
        "X-Multi-Word": "value with spaces and = signs",
        "X-UTF8": "héllo-wörld",
        "X-Empty": "",
    }
    for name, value in payload.items():
        (tmp_path / name).write_text(value)

    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(tmp_path)) == payload


def test_skips_kubelet_dotfile_atomic_update_symlinks(tmp_path):
    """Kubelet writes a `..data` symlink + `..2026_05_07...` directory for atomic updates.

    Those entries start with `.` and must be filtered out so they don't show up
    as bogus headers.
    """
    (tmp_path / "Authorization").write_text("Bearer token-xyz")
    # Mimic kubelet's atomic-update artifacts
    (tmp_path / "..data").write_text("not-a-header")
    (tmp_path / "..2026_05_07_12_34_56.789").mkdir()

    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(tmp_path)) == {
        "Authorization": "Bearer token-xyz",
    }


def test_skips_subdirectories(tmp_path):
    """Only regular files become headers; subdirectories are ignored."""
    (tmp_path / "Authorization").write_text("Bearer token-xyz")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "X-Nested").write_text("ignored")

    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(tmp_path)) == {
        "Authorization": "Bearer token-xyz",
    }


def test_returns_empty_and_logs_warning_on_oserror(monkeypatch, tmp_path):
    """If scandir/open raises OSError mid-read, fail-open and log."""

    def boom(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(mlrun.utils.telemetry.os, "scandir", boom)

    warn_spy = MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warn_spy)

    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(tmp_path)) == {}

    warn_spy.assert_called_once()
    args, kwargs = warn_spy.call_args
    assert "Failed to read OTLP telemetry headers from mount" in args[0]
    assert kwargs.get("path") == str(tmp_path)


def test_logs_debug_when_headers_resolved(tmp_path, monkeypatch):
    (tmp_path / "Authorization").write_text("Bearer token-xyz")
    debug_spy = MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "debug", debug_spy)

    mlrun.utils.telemetry.resolve_otlp_headers(path=str(tmp_path))

    debug_spy.assert_called_once()
    args, kwargs = debug_spy.call_args
    assert "Resolved OTLP telemetry headers from mount" in args[0]
    assert kwargs.get("header_keys") == ["Authorization"]


def test_does_not_log_debug_when_directory_is_empty(tmp_path, monkeypatch):
    """If the directory exists but is empty, we shouldn't claim we resolved anything."""
    debug_spy = MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "debug", debug_spy)

    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(tmp_path)) == {}

    debug_spy.assert_not_called()


def test_path_is_a_file_not_a_directory(tmp_path):
    """If the configured path exists but is a file (misconfiguration), return {}."""
    not_a_dir = tmp_path / "headers-file"
    not_a_dir.write_text("oops")

    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(not_a_dir)) == {}


def test_default_path_is_the_constant(monkeypatch, tmp_path):
    """When no path is passed, the helper falls back to MLRUN_TELEMETRY_OTLP_HEADERS_PATH."""
    (tmp_path / "Authorization").write_text("Bearer abc")
    monkeypatch.setattr(
        mlrun.common.constants,
        "MLRUN_TELEMETRY_OTLP_HEADERS_PATH",
        str(tmp_path),
    )

    assert mlrun.utils.telemetry.resolve_otlp_headers() == {
        "Authorization": "Bearer abc"
    }


def test_explicit_path_overrides_default(monkeypatch, tmp_path):
    """Explicit `path` argument wins over MLRUN_TELEMETRY_OTLP_HEADERS_PATH."""
    default_dir = tmp_path / "default"
    explicit_dir = tmp_path / "explicit"
    default_dir.mkdir()
    explicit_dir.mkdir()
    (default_dir / "Authorization").write_text("from-default")
    (explicit_dir / "Authorization").write_text("from-explicit")

    monkeypatch.setattr(
        mlrun.common.constants,
        "MLRUN_TELEMETRY_OTLP_HEADERS_PATH",
        str(default_dir),
    )

    # Sanity: default produces "from-default"
    assert mlrun.utils.telemetry.resolve_otlp_headers() == {
        "Authorization": "from-default"
    }
    # Explicit path wins
    assert mlrun.utils.telemetry.resolve_otlp_headers(path=str(explicit_dir)) == {
        "Authorization": "from-explicit"
    }
