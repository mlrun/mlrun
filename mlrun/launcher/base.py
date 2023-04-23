# Copyright 2023 MLRun Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import ast
import getpass
from copy import deepcopy
from os import environ
from typing import Any, Dict, List, Optional, Union

import mlrun.model
from mlrun.api import schemas
from mlrun.model import RunObject, RunTemplate
from mlrun.runtimes import BaseRuntime

run_modes = ["pass"]


class BaseLauncher(abc.ABC):
    """
    Abstract class for managing and running functions in different contexts
    This class is designed to encapsulate the logic of running a function in different contexts
    i.e. running a function locally, remotely or in a server
    Each context will have its own implementation of the abstract methods
    """

    @staticmethod
    @abc.abstractmethod
    def verify_base_image(runtime):
        """resolves and sets the build base image if build is needed"""
        pass

    @staticmethod
    @abc.abstractmethod
    def save(runtime):
        """store the function to the db"""
        pass

    def run(
        self,
        runtime: BaseRuntime,
        runspec: Union[str, dict, RunTemplate, RunObject] = None,
        handler=None,
        name: str = "",
        project: str = "",
        params: dict = None,
        inputs: Dict[str, str] = None,
        out_path: str = "",
        workdir: str = "",
        artifact_path: str = "",
        watch: bool = True,
        schedule: Union[str, schemas.ScheduleCronTrigger] = None,
        hyperparams: Dict[str, list] = None,
        hyper_param_options: mlrun.model.HyperParamOptions = None,
        verbose=None,
        scrape_metrics: bool = None,
        local=False,
        local_code_path=None,
        auto_build=None,
        param_file_secrets: Dict[str, str] = None,
        notifications: List[mlrun.model.Notification] = None,
        returns: Optional[List[Union[str, Dict[str, str]]]] = None,
    ) -> RunObject:
        """
        Run the function from the server/client[local/remote]

        :param runtime:        runtime object which runs the function by triggering the launcher
        :param runspec:        run template object or dict (see RunTemplate)
        :param handler:        pointer or name of a function handler
        :param name:           execution name
        :param project:        project name
        :param params:         input parameters (dict)
        :param inputs:         Input objects to pass to the handler. Type hints can be given so the input will be parsed
                               during runtime from `mlrun.DataItem` to the given type hint. The type hint can be given
                               in the key field of the dictionary after a colon, e.g: "<key> : <type_hint>".
        :param out_path:       default artifact output path
        :param artifact_path:  default artifact output path (will replace out_path)
        :param workdir:        default input artifacts path
        :param watch:          watch/follow run log
        :param schedule:       ScheduleCronTrigger class instance or a standard crontab expression string
                               (which will be converted to the class using its `from_crontab` constructor),
                               see this link for help:
                               https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html#module-apscheduler.triggers.cron
        :param hyperparams:    dict of param name and list of values to be enumerated e.g. {"p1": [1,2,3]}
                               the default strategy is grid search, can specify strategy (grid, list, random)
                               and other options in the hyper_param_options parameter
        :param hyper_param_options:  dict or :py:class:`~mlrun.model.HyperParamOptions` struct of
                                     hyper parameter options
        :param verbose:        add verbose prints/logs
        :param scrape_metrics: whether to add the `mlrun/scrape-metrics` label to this run's resources
        :param local:      run the function locally vs on the runtime/cluster
        :param local_code_path: path of the code for local runs & debug
        :param auto_build: when set to True and the function require build it will be built on the first
                           function run, use only if you dont plan on changing the build config between runs
        :param param_file_secrets: dictionary of secrets to be used only for accessing the hyper-param parameter file.
                            These secrets are only used locally and will not be stored anywhere
        :param notifications: list of notifications to push when the run is completed
        :param returns: List of log hints - configurations for how to log the returning values from the handler's run
                        (as artifacts or results). The list's length must be equal to the amount of returning objects. A
                        log hint may be given as:

                        * A string of the key to use to log the returning value as result or as an artifact. To specify
                          The artifact type, it is possible to pass a string in the following structure:
                          "<key> : <type>". Available artifact types can be seen in `mlrun.ArtifactType`. If no
                          artifact type is specified, the object's default artifact type will be used.
                        * A dictionary of configurations to use when logging. Further info per object type and artifact
                          type can be given there. The artifact key must appear in the dictionary as "key": "the_key".

        :return: run context object (RunObject) with run metadata, results and status
        """
        self._validate_runtime(runtime, inputs)
        # TODO: Needed for client side
        # self._enrich_function(runtime)
        run = self._create_run_object(runspec)
        return self._run(runtime, run, param_file_secrets)

    @abc.abstractmethod
    def _run(self, runtime: BaseRuntime, run: RunObject, param_file_secrets):
        raise NotImplementedError()

    @staticmethod
    def _create_run_object(run: Union[str, dict, RunTemplate, RunObject]) -> RunObject:
        # TODO: Once implemented the `Runtime` handlers configurations (doc strings, params type hints and returning
        #       log hints, possible parameter values, etc), the configured type hints and log hints should be set into
        #       the `RunObject` from the `Runtime`.
        if run:
            run = deepcopy(run)
            if isinstance(run, str):
                run = ast.literal_eval(run)
            if not isinstance(run, (dict, RunTemplate, RunObject)):
                raise ValueError(
                    "task/runspec is not a valid task object," f" type={type(run)}"
                )

        if isinstance(run, RunTemplate):
            run = RunObject.from_template(run)
        if isinstance(run, dict) or run is None:
            run = RunObject.from_dict(run)
        return run

    @staticmethod
    @abc.abstractmethod
    def _enrich_run(runtime: BaseRuntime, run: RunObject):
        pass

    @staticmethod
    def _store_function(runtime: BaseRuntime, run: RunObject, db):
        run.metadata.labels["kind"] = runtime.kind
        if "owner" not in run.metadata.labels:
            run.metadata.labels["owner"] = (
                environ.get("V3IO_USERNAME") or getpass.getuser()
            )
        if run.spec.output_path:
            run.spec.output_path = run.spec.output_path.replace(
                "{{run.user}}", run.metadata.labels["owner"]
            )

        if db and runtime.kind != "handler":
            struct = runtime.to_dict()
            hash_key = db.store_function(
                struct, runtime.metadata.name, runtime.metadata.project, versioned=True
            )
            run.spec.function = runtime._function_uri(hash_key=hash_key)

    @staticmethod
    def _validate_runtime(
        runtime: BaseRuntime,
        inputs: Dict[str, str] = None,
    ):
        mlrun.utils.helpers.verify_dict_items_type("Inputs", inputs, [str], [str])

        if runtime.spec.mode and runtime.spec.mode not in run_modes:
            raise ValueError(f'run mode can only be {",".join(run_modes)}')

    def _verify_run_params(self, parameters: Dict[str, Any]):
        for param_name, param_value in parameters.items():

            if isinstance(param_value, dict):
                # if the parameter is a dict, we might have some nested parameters,
                # in this case we need to verify them as well recursively
                self._verify_run_params(param_value)

            # verify that integer parameters don't exceed a int64
            if isinstance(param_value, int) and abs(param_value) >= 2**63:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"parameter {param_name} value {param_value} exceeds int64"
                )
