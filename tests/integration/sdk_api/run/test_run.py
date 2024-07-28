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
import mlrun
import mlrun.common.constants as mlrun_constants
import tests.integration.sdk_api.base


class TestRun(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_ctx_creation_creates_run_with_project(self):
        ctx_name = "some-context"
        project = mlrun.new_project(mlrun.mlconf.default_project)
        project.save()

        mlrun.get_or_create_ctx(ctx_name)
        runs = mlrun.get_run_db().list_runs(
            name=ctx_name, project=mlrun.mlconf.default_project
        )
        assert len(runs) == 1
        assert runs[0]["metadata"]["project"] == mlrun.mlconf.default_project

    def test_ctx_state_change(self):
        ctx_name = "some-context"
        project = mlrun.new_project(mlrun.mlconf.default_project)
        project.save()

        ctx = mlrun.get_or_create_ctx(ctx_name)
        runs = mlrun.get_run_db().list_runs(
            name=ctx_name, project=mlrun.mlconf.default_project
        )
        assert len(runs) == 1
        assert (
            runs[0]["status"]["state"]
            == mlrun.common.runtimes.constants.RunStates.running
        )
        ctx.set_state(mlrun.common.runtimes.constants.RunStates.completed)
        runs = mlrun.get_run_db().list_runs(
            name=ctx_name, project=mlrun.mlconf.default_project
        )
        assert len(runs) == 1
        assert (
            runs[0]["status"]["state"]
            == mlrun.common.runtimes.constants.RunStates.completed
        )

    def test_ctx_run_labels(self):
        ctx_name = "some-context"
        project = mlrun.new_project(mlrun.mlconf.default_project)
        project.save()

        ctx = mlrun.get_or_create_ctx(ctx_name)
        runs = mlrun.get_run_db().list_runs(
            name=ctx_name, project=mlrun.mlconf.default_project
        )
        assert len(runs) == 1

        _remove_internal_labels(runs)

        assert runs[0]["metadata"]["labels"] == {}

        ctx.set_label("label-key", "label-value")
        ctx._update_run(commit=True)
        runs = mlrun.get_run_db().list_runs(
            name=ctx_name, project=mlrun.mlconf.default_project
        )
        assert len(runs) == 1

        _remove_internal_labels(runs)

        assert runs[0]["metadata"]["labels"] == {"label-key": "label-value"}

        # mock not logging worker
        ctx.set_label(mlrun_constants.MLRunInternalLabels.host, "worker-1")
        ctx.set_label("kind", "mpijob")
        assert not ctx.is_logging_worker()
        ctx._update_run(commit=True)

        # labels should remain the same
        runs = mlrun.get_run_db().list_runs(
            name=ctx_name, project=mlrun.mlconf.default_project
        )
        assert len(runs) == 1

        _remove_internal_labels(runs)

        assert runs[0]["metadata"]["labels"] == {"label-key": "label-value"}


def _remove_internal_labels(runs):
    assert mlrun_constants.MLRunInternalLabels.host in runs[0]["metadata"]["labels"]
    del runs[0]["metadata"]["labels"][mlrun_constants.MLRunInternalLabels.host]
    assert mlrun_constants.MLRunInternalLabels.kind in runs[0]["metadata"]["labels"]
    del runs[0]["metadata"]["labels"][mlrun_constants.MLRunInternalLabels.kind]
