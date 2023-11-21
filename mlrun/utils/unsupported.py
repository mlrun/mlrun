# Copyright 2023 Iguazio
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

from contextlib import contextmanager
from importlib import import_module
from inspect import isclass
from unittest import mock

from mlrun.errors import err_to_str
from mlrun.utils import logger

dependencies = [
    "kfp.dsl.ParallelFor",
]


@contextmanager
def disable_unsupported_external_features():
    """
    Disables functionality external to MLRun that is known to cause problems during workflow parsing
    This function operates on a best-effort basis and lets errors pass mostly silently
    Only features based on classes are supported since this patches their __init__
    """
    try:
        originals = []
        for d in dependencies:
            module_name, _, callable_name = d.rpartition(".")
            try:
                curr = getattr(import_module(module_name), callable_name)
            except (ImportError, KeyError) as exc:
                logger.debug(
                    "Module import for dependency patching failed",
                    dependency=d,
                    exc_info=err_to_str(exc),
                )
                continue

            if not isclass(curr):
                logger.debug(
                    "MLRun only supports patching of features based on classes",
                    dependency=d,
                )
                continue

            originals.append(
                {
                    "original_method": curr.__init__,
                    "class": curr,
                }
            )
            curr.__init__ = mock.MagicMock(
                side_effect=NotImplementedError(f"MLRun does not support {d}")
            )
        # Signal the caller that the context is ready
        yield
    except Exception as exc:
        logger.warning(
            "Error while trying to disable external features not supported by MLRun; unexpected behaviour may occur",
            exc_info=err_to_str(exc),
        )
        # Tentatively, allow user to proceed
        yield
    finally:
        # When caller signals for the closure of the context, we revert all patches
        for o in originals:
            try:
                o["class"].__init__ = o["original_method"]
            except Exception as exc:
                logger.warning(
                    "Error while trying to re-enable an external feature not supported by MLRun",
                    exc_info=err_to_str(exc),
                )
                # Continue trying to re-enable other external features
                continue
