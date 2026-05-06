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

import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestJobStoreUri(tests.system.base.TestMLRunSystem):
    """End-to-end coverage for kind=job functions whose source is a store://
    CodeArtifact URI.

    pr-b (#9609) shipped the SDK + server-side enrich pieces for this flow
    but had no end-to-end test. This file fills that gap and verifies the
    three colon-handler fixes that landed alongside it (launcher kind
    selector, LocalRuntime._pre_run command-pointing, auto-name separator)
    by using the canonical ``module:function`` handler form.
    """

    project_name = "job-store-uri-system-test"

    def custom_setup(self):
        super().custom_setup()
        self._handler_filename = "handler.py"
        self._function_image = os.environ.get("MLRUN_TEST_IMAGE", "mlrun/mlrun")

    def _log_code_artifact(self, key: str) -> str:
        """Log the assets/handler.py file as a CodeArtifact and return its
        canonical store:// URI.

        The asset (``tests/system/runtimes/assets/handler.py``) is the
        existing handler used by other system tests in this directory —
        its ``my_func`` callable returns deterministic outputs we assert on.

        Uses the artifact's own ``.uri`` rather than rebuilding
        ``f"store://artifacts/<project>/<key>"`` from a template — the
        reconstructed value can drift from what the system actually stored
        (tag suffixes, scheme changes). See TESTING_STANDARDS §7.
        """
        local_path = os.path.join(self.assets_path, self._handler_filename)
        artifact = self.project.log_code_file(key=key, local_path=local_path)
        return artifact.uri

    def test_e2e_job_function_from_store_artifact(self):
        """Job kind + store:// + canonical ``module:function`` handler form.

        Flow:
          1. log_code_file uploads handler.py as a CodeArtifact and returns
             its store:// URI.
          2. set_function with kind="job", handler="handler:my_func".
          3. function.run(params={"p1": 5}) — the pod downloads the artifact
             at startup, _pre_run points spec.command at the extracted
             handler.py, and my_func runs.
          4. Asserts state=completed and accuracy=10 (my_func sets
             accuracy = p1 * 2).
        """
        store_uri = self._log_code_artifact("job_code")

        function = self.project.set_function(
            func=store_uri,
            name="job-from-store",
            kind="job",
            handler="handler:my_func",
            image=self._function_image,
        )

        run = function.run(params={"p1": 5})

        assert run.status.state == "completed", (
            f"Run did not complete: state={run.status.state}, error={run.status.error}"
        )
        assert run.status.results["accuracy"] == 10, (
            f"Expected accuracy=10, got {run.status.results.get('accuracy')!r}"
        )

        # store:// must remain in the function spec after run - server must
        # never resolve it back to s3:// in the DB.
        db_function = self.project.get_function("job-from-store")
        assert db_function.spec.build.source == store_uri, (
            f"Expected store:// URI preserved in DB, got "
            f"{db_function.spec.build.source!r}"
        )
