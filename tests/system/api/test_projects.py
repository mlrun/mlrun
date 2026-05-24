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

import uuid

import pytest

import mlrun
import mlrun.common.schemas
import mlrun.errors
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestStoreProjectOPARace(TestMLRunSystem):
    # The test only exercises the mlrun control plane (create / save project);
    # no V3IO data plane is touched, so we deliberately omit
    # ``@pytest.mark.enterprise`` to keep mandatory env vars minimal
    # (``MLRUN_DBPATH`` only).
    """
    End-to-end coverage of the OPA-manifest-propagation race on
    PUT ``/projects/{name}``: a freshly-created project's manifest may not
    yet have reached every API replica's OPA sidecar, and an immediate
    follow-up ``store_project`` would otherwise return 403.

    Each iteration creates a uniquely-named project and immediately stores
    it again, which is the exact pattern the SDK uses in
    ``get_or_create_project`` followed by ``project.save()``. Multiple
    iterations give the race more chances to manifest on a multi-replica
    deployment.

    Pre-fix expectation (with the airun-side retry temporarily removed in
    parallel): at least one iteration trips
    ``MLRunAccessDeniedError`` for ``/resources/projects/<name>``.
    Post-fix expectation: all iterations complete cleanly.
    """

    project_name = "system-test-store-project-opa-race"
    _skip_set_environment = lambda self: True  # noqa: E731

    def custom_setup(self):
        # NB: the fix targets ``iguazio_v4`` mode, where the projects endpoint
        # consults OPA on every store_project. In IG3/CE the same code path
        # early-returns for leader-originated requests, so this test still
        # passes there — vacuously, but it remains a valid regression guard
        # for "create-then-store must not 403" on any backend.
        self.created_project_names: list[str] = []

    def custom_teardown(self):
        for name in self.created_project_names:
            try:
                self._run_db.delete_project(
                    name,
                    deletion_strategy=mlrun.common.schemas.DeletionStrategy.cascading,
                )
            except mlrun.errors.MLRunNotFoundError:
                pass

    @pytest.mark.parametrize("iteration", range(5))
    def test_create_then_immediate_store_does_not_403(self, iteration: int):
        name = f"sys-opa-race-{uuid.uuid4().hex[:8]}"
        self.created_project_names.append(name)

        project = mlrun.get_or_create_project(
            name, context="./", user_project=False, allow_cross_project=True
        )

        # The race window: create returned, OPA bundle may not have propagated
        # to all API pods yet, immediate store_project follow-up must succeed.
        try:
            project.save()
        except mlrun.errors.MLRunAccessDeniedError as exc:
            pytest.fail(
                f"store_project hit OPA-cache race window on iteration {iteration}: {exc}. "
                "This is the regression fixed by the projects.py + opa.py changes."
            )
