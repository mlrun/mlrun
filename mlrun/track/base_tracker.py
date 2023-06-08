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

import abc
import os
import pathlib
import typing
from mlrun.config import config as mlconf
from .tracker import Tracker

from mlrun.execution import MLClientCtx


class BaseTracker(Tracker):
    MODULE_NAME = ...

    def is_enabled(self) -> bool:
        """
        Checks if tracker is enabled. only checks in config.

        Returns:
            bool: True if the feature is enabled, False otherwise.
        """
        return mlconf.trackers.is_enabled

    def pre_run(self, context: MLClientCtx, env: dict, args: dict) -> (dict, dict):
        """
        Initializes the tracking system for a 3rd party module.

        This function sets up the necessary components and resources to enable tracking of params, artifacts,
        or metrics within the module.
        Returns:
            two dictionaries, env and args containing environment and tracking data
        """
        env = {}
        experiment = self._tracked_platform.get_experiment_by_name(context.name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = self._tracked_platform.create_experiment(context.name)
        env[
            f"{self.MODULE_NAME.upper()}_RUN_CONTEXT"
        ] = '{"mlrun_runid": "%s", "mlrun_project": "%s"}' % (
            context._uid,
            context.project,
        )
        env[f"{self.MODULE_NAME.upper()}_EXPERIMENT_ID"] = experiment_id
        args[f"{self.MODULE_NAME}_experiment"] = experiment_id
        return env, args

    def _post_run(
        self, context: typing.Union[MLClientCtx, dict], args: dict, experiment: str = None,
    ):
        """
        !!! not to confuse with post_run!!!
        Performs post-run tasks of logging 3rd party artifacts generated during the run.
        """
        experiment_id = args.get(experiment)
        runs = self._client.search_runs(
            experiment_id, filter_string=f'tags.mlrun_runid="{context._uid}"'
        )
        if not runs:
            experiments = [
                experiment.experiment_id
                for experiment in self._client.search_experiments()
            ]
            runs = self._client.search_runs(
                experiments, filter_string=f'tags.mlrun_runid="{context._uid}"'
            )

        if runs:
            for run in runs:  # each run gets logged in a different child process
                child_context = context.get_child_context(with_parent_params=True)
                self._log_run(child_context, run)
            context.commit()

    def _log_run(self, context, run):
        '''
        after mlrun function's run is done, copy all data logged by third party app tracker

        '''
        self._artifacts = {}
        for key, val in run.data.params.items():
            context._parameters[key] = val
        context.log_results(run.data.metrics)
        context.set_label(f"{self.MODULE_NAME}-runid", run.info.run_id)
        context.set_label(f"{self.MODULE_NAME}-experiment", run.info.experiment_id)
        for artifact in self._client.list_artifacts(run.info.run_id):
            full_path = self._tracked_platform.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path=artifact.path
            )
            if artifact.is_dir and os.path.exists(os.path.join(full_path, "MLmodel")):
                model_path = full_path  # this is the model folder, we log it after logging all artifacts
            else:
                artifact = context.log_artifact(
                    item=pathlib.Path(artifact.path).stem, local_path=full_path
                )
                self._artifacts[artifact.key] = artifact

            self.log_model(model_path, context)

    @abc.abstractmethod
    def log_model(self, model_uri, context):
        pass

    @abc.abstractmethod
    def post_run(self, model_uri, context):
        pass
