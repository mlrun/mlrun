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

import pathlib
import typing

import pytest

import mlrun


@pytest.fixture
def new_project_factory(
    tmp_path: pathlib.Path,
) -> typing.Callable[..., mlrun.projects.project.MlrunProject]:
    """
    Create MLRun projects with an isolated filesystem context.

    This prevents test contamination from using the default project context ("./"),
    which depends on the current working directory and may leak state between tests.
    """

    def _new_project(name: str, **kwargs) -> mlrun.projects.project.MlrunProject:
        # only set context if it is not already set
        context = kwargs.pop("context", None)
        if context is None:
            context_path = tmp_path / "projects" / name
            context_path.mkdir(parents=True, exist_ok=True)
            context = context_path.as_posix()
        return mlrun.new_project(name, context=context, **kwargs)

    return _new_project
