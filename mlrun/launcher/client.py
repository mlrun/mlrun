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
import getpass
import os
from typing import Optional

import IPython

import mlrun.errors
import mlrun.launcher.base as launcher
import mlrun.lists
import mlrun.model
import mlrun.runtimes
from mlrun.utils import logger


class ClientBaseLauncher(launcher.BaseLauncher, abc.ABC):
    """
    Abstract class for common code between client launchers
    """

    def enrich_runtime(
        self,
        runtime: "mlrun.runtimes.base.BaseRuntime",
        project_name: Optional[str] = "",
        full: bool = True,
    ):
        runtime.try_auto_mount_based_on_config()
        runtime._fill_credentials()

    @staticmethod
    def prepare_image_for_deploy(runtime: "mlrun.runtimes.BaseRuntime"):
        """
        Check if the runtime requires to build the image.
        If build is needed, set the image as the base_image for the build.
        If image is not given set the default one.
        """
        if runtime.kind in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
            return

        build = runtime.spec.build
        require_build = (
            build.commands
            or build.requirements
            or (build.source and not build.load_source_on_run)
        )
        image = runtime.spec.image
        # we allow users to not set an image, in that case we'll use the default
        if (
            not image
            and runtime.kind in mlrun.mlconf.function_defaults.image_by_kind.to_dict()
        ):
            image = mlrun.mlconf.function_defaults.image_by_kind.to_dict()[runtime.kind]

        # TODO: need a better way to decide whether a function requires a build
        if require_build and image and not runtime.spec.build.base_image:
            # when the function require build use the image as the base_image for the build
            runtime.spec.build.base_image = image
            runtime.spec.image = ""

    @staticmethod
    def _store_function(
        runtime: "mlrun.runtimes.BaseRuntime", run: "mlrun.run.RunObject"
    ):
        run.metadata.labels["kind"] = runtime.kind
        if "owner" not in run.metadata.labels:
            run.metadata.labels["owner"] = (
                os.environ.get("V3IO_USERNAME") or getpass.getuser()
            )
        if run.spec.output_path:
            run.spec.output_path = run.spec.output_path.replace(
                "{{run.user}}", run.metadata.labels["owner"]
            )
        db = runtime._get_db()
        if db and runtime.kind != "handler":
            struct = runtime.to_dict()
            hash_key = db.store_function(
                struct, runtime.metadata.name, runtime.metadata.project, versioned=True
            )
            run.spec.function = runtime._function_uri(hash_key=hash_key)

    @staticmethod
    def _refresh_function_metadata(runtime: "mlrun.runtimes.BaseRuntime"):
        try:
            meta = runtime.metadata
            db = runtime._get_db()
            db_func = db.get_function(meta.name, meta.project, meta.tag)
            if db_func and "status" in db_func:
                runtime.status = db_func["status"]
                if (
                    runtime.status.state
                    and runtime.status.state == "ready"
                    and runtime.kind
                    # We don't want to override the nuclio image here because the build happens in nuclio
                    # TODO: have a better way to check if nuclio function deploy started
                    and not hasattr(runtime.status, "nuclio_name")
                ):
                    runtime.spec.image = mlrun.utils.get_in(
                        db_func, "spec.image", runtime.spec.image
                    )
        except mlrun.errors.MLRunNotFoundError:
            pass

    @staticmethod
    def _log_track_results(
        runtime: "mlrun.runtimes.BaseRuntime", result: dict, run: "mlrun.run.RunObject"
    ):
        """
        log commands to track results
        in jupyter, displays a table widget with the result
        else, logs CLI commands to track results and a link to the results in UI

        :param: runtime: runtime object
        :param result:   run result dict
        :param run:      run object
        """
        uid = run.metadata.uid
        project = run.metadata.project

        # show ipython/jupyter result table widget
        results_tbl = mlrun.lists.RunList()
        if result:
            results_tbl.append(result)
        else:
            logger.info("no returned result (job may still be in progress)")
            results_tbl.append(run.to_dict())

        if mlrun.utils.is_ipython and mlrun.config.config.ipython_widget:
            results_tbl.show()
            print()
            ui_url = mlrun.utils.get_ui_url(project, uid)
            if ui_url:
                ui_url = f' or <a href="{ui_url}" target="_blank">click here</a> to open in UI'
            IPython.display.display(
                IPython.display.HTML(
                    f"<b> > to track results use the .show() or .logs() methods {ui_url}</b>"
                )
            )
        elif not runtime.is_child:
            # TODO: Log sdk commands to track results instead of CLI commands
            project_flag = f"-p {project}" if project else ""
            info_cmd = f"mlrun get run {uid} {project_flag}"
            logs_cmd = f"mlrun logs {uid} {project_flag}"
            logger.info(
                "To track results use the CLI", info_cmd=info_cmd, logs_cmd=logs_cmd
            )
            ui_url = mlrun.utils.get_ui_url(project, uid)
            if ui_url:
                logger.info("Or click for UI", ui_url=ui_url)
