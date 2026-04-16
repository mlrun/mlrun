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

import os
import traceback

_real_os_exit = os._exit

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_COVERAGE_ERROR_LOG_DIR = os.path.join(_ROOT_DIR, "tests", "coverage_reports", "errors")


def _coverage_saving_exit(status):
    """Save coverage data before os._exit() in a forked child."""
    status = status or 0
    try:
        import coverage

        current_coverage = coverage.Coverage.current()

        if current_coverage is None:
            raise RuntimeError(
                "COVERAGE_PROCESS_START is set but no active Coverage instance "
                "found in forked child - coverage data will be lost."
            )
        current_coverage.stop()
        current_coverage.save()
    except Exception:
        log_path = os.path.join(
            _COVERAGE_ERROR_LOG_DIR, f"coverage_error_{os.getpid()}.log"
        )
        with open(log_path, "a") as f:
            f.write(traceback.format_exc())
        status = 1
    finally:
        _real_os_exit(status)


def _patch_exit_for_coverage():
    os._exit = _coverage_saving_exit


if os.environ.get("COVERAGE_PROCESS_START"):
    os.register_at_fork(after_in_child=_patch_exit_for_coverage)
