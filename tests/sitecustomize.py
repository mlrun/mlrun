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

_real_os_exit = os._exit


def _coverage_saving_exit(status):
    """Save coverage data before os._exit() in a forked child."""
    try:
        import coverage

        cov = coverage.Coverage.current()
        if cov is not None:
            cov.stop()
            cov.save()
    except Exception:
        pass
    _real_os_exit(status)


def _patch_exit_for_coverage():
    os._exit = _coverage_saving_exit


if os.environ.get("COVERAGE_PROCESS_START"):
    os.register_at_fork(after_in_child=_patch_exit_for_coverage)
