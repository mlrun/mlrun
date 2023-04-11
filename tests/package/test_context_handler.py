# Copyright 2018 Iguazio
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
#
from types import FunctionType

import pytest

import mlrun
from mlrun import MLClientCtx
from mlrun.package import ContextHandler


def test_init():
    """
    During the context handler's initialization, it collects the default packagers found in the class variables
    `_MLRUN_REQUIREMENTS_PACKAGERS`, `_EXTENDED_PACKAGERS` and `_MLRUN_FRAMEWORKS_PACKAGERS` so this test is making sure
    there is no error raised during the init collection of packagers when new ones are being added.
    """
    ContextHandler()


def _look_for_context_via_get_or_create(not_a_context=None):
    assert not isinstance(not_a_context, MLClientCtx)
    context_handler = ContextHandler()
    context_handler.look_for_context(args=(), kwargs={})
    return context_handler.is_context_available()


def _look_for_context_via_header(context: MLClientCtx):
    context_handler = ContextHandler()
    context_handler.look_for_context(args=(), kwargs={"context": context})
    return context_handler.is_context_available()


@pytest.mark.parametrize(
    "func",
    [_look_for_context_via_get_or_create, _look_for_context_via_header],
)
@pytest.mark.parametrize("result", [True, False])
def test_look_for_context(rundb_mock, func: FunctionType, result: bool):
    """
    Test the `look_for_context` method of the context handler. The method should find or create a context only when it
    is being ran through MLRun.

    :param rundb_mock: A runDB mock fixture.
    :param func:       The function to run in the test.
    :param result:     Boolean flag to expect to find a context (run via MLRun) as True and to not find a context as
                       False.
    """
    if result:
        assert not func(None)
        return
    run = mlrun.new_function().run(handler=func, returns=["result:result"])
    assert run.status.results["result"]


# TODO: finish test.
def test_custom_packagers(rundb_mock):
    """
    Test the custom packagers collection from a project during the `look_for_context` method.

    :param rundb_mock: A runDB mock fixture.
    """
    pass
