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
#
from types import FunctionType

import pytest

import mlrun
from mlrun import MLClientCtx
from mlrun.package import ContextHandler
from mlrun.runtimes import RunError


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
@pytest.mark.parametrize("run_via_mlrun", [True, False])
def test_look_for_context(rundb_mock, func: FunctionType, run_via_mlrun: bool):
    """
    Test the `look_for_context` method of the context handler. The method should find or create a context only when it
    is being run through MLRun.

    :param rundb_mock:        A runDB mock fixture.
    :param func:              The function to run in the test.
    :param run_via_mlrun:     Boolean flag to expect to find a context (run via MLRun) as True and to not find a context
                              as False.
    """
    if not run_via_mlrun:
        assert not func(None)
        return
    run = mlrun.new_function().run(handler=func, returns=["result:result"])
    assert run.status.results["result"]


def collect_custom_packagers():
    return


@pytest.mark.parametrize(
    "packager, expected_result",
    [
        ("tests.package.test_packagers_manager.PackagerA", True),
        ("tests.package.packagers_testers.default_packager_tester.SomeClass", False),
    ],
)
@pytest.mark.parametrize("is_mandatory", [True, False])
def test_custom_packagers(
    rundb_mock, packager: str, expected_result: bool, is_mandatory: bool
):
    """
    Test the custom packagers collection from a project during the `look_for_context` method.

    :param rundb_mock:      A runDB mock fixture.
    :param packager:        The custom packager to collect.
    :param expected_result: Whether the packager collection should succeed.
    :param is_mandatory:    If the packager is mandatory for the run or not. Mandatory packagers will always raise
                            exception if they couldn't be collected.
    """
    project = mlrun.get_or_create_project(name="default", allow_cross_project=True)
    project.add_custom_packager(
        packager=packager,
        is_mandatory=is_mandatory,
    )
    project.save_to_db()
    mlrun_function = project.set_function(
        func=__file__, name="test_custom_packagers", image="mlrun/mlrun"
    )
    if expected_result or not is_mandatory:
        mlrun_function.run(handler="collect_custom_packagers", local=True)
        return
    try:
        mlrun_function.run(handler="collect_custom_packagers", local=True)
        assert False
    except RunError:
        pass
