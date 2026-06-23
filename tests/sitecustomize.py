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

import glob as _glob
import os
import time
import traceback

_real_os_exit = os._exit

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_COVERAGE_ERROR_LOG_DIR = os.path.join(_ROOT_DIR, "tests", "coverage_reports", "errors")

_DIAG_LOG_DIR = os.path.join(_ROOT_DIR, "tests", "coverage_reports", "diagnostics")
_DIAG_LOG_PATH = os.path.join(_DIAG_LOG_DIR, "fork_diagnostic.log")


def _diag_log(tag, **fields):
    try:
        os.makedirs(_DIAG_LOG_DIR, exist_ok=True)
        parts = [f"ts={time.time():.3f}", f"pid={os.getpid()}", f"ppid={os.getppid()}", f"tag={tag}"]
        for k, v in fields.items():
            parts.append(f"{k}={v}")
        with open(_DIAG_LOG_PATH, "a") as f:
            f.write(" ".join(parts) + "\n")
    except Exception:
        pass


def _coverage_saving_exit(status):
    """Save coverage data before os._exit() in a forked child."""
    status = status or 0
    try:
        import coverage

        current_coverage = coverage.Coverage.current()

        if current_coverage is None:
            _diag_log("exit_no_current_coverage")
            raise RuntimeError(
                "COVERAGE_PROCESS_START is set but no active Coverage instance "
                "found in forked child - coverage data will be lost."
            )
        cov_file_env = os.environ.get("COVERAGE_FILE", "")
        _diag_log("exit_pre_save", cov_file_env=cov_file_env)
        current_coverage.stop()
        current_coverage.save()
        if cov_file_env:
            matches = sorted(_glob.glob(cov_file_env + ".*"))
            _diag_log("exit_post_save", matches_count=len(matches), matches="|".join(matches))
        else:
            _diag_log("exit_post_save_no_cov_file_env")
    except Exception as exc:
        log_path = os.path.join(
            _COVERAGE_ERROR_LOG_DIR, f"coverage_error_{os.getpid()}.log"
        )
        os.makedirs(_COVERAGE_ERROR_LOG_DIR, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(traceback.format_exc())
        _diag_log("exit_exception", err=type(exc).__name__)
        status = 1
    finally:
        _real_os_exit(status)


def _patch_exit_for_coverage():
    _diag_log("after_in_child_fired")
    os._exit = _coverage_saving_exit


if os.environ.get("COVERAGE_PROCESS_START"):
    _diag_log("sitecustomize_registered")
    os.register_at_fork(after_in_child=_patch_exit_for_coverage)
