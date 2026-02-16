# Copyright 2023 Iguazio
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
import base64
import gzip
from copy import deepcopy
from typing import Optional, Union

from dependency_injector import containers, providers

import mlrun.auth.utils
import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes.constants
import mlrun.common.schemas.schedule
import mlrun.config
import mlrun.execution
import mlrun.k8s_utils
import mlrun.launcher.base as launcher
import mlrun.launcher.factory
import mlrun.projects.operations
import mlrun.projects.pipelines
import mlrun.runtimes
import mlrun.runtimes.generators
import mlrun.runtimes.utils
import mlrun.utils
import mlrun.utils.helpers
import mlrun.utils.regex
from mlrun.model import RunSpec, RunTemplate
from mlrun.runtimes import KubejobRuntime, RemoteRuntime

import framework.api.utils
import framework.db.session
import framework.utils.helpers
import framework.utils.singletons.db
import services.api.crud
import services.api.runtime_handlers
import services.api.utils.helpers

# Configmap objects on Kubernetes have 10Mb size limit
SERVING_SPEC_MAX_LENGTH = 10485760


class ServerSideLauncher(launcher.BaseLauncher):
    def __init__(
        self,
        local: bool = False,
        auth_info: Optional[mlrun.common.schemas.AuthInfo] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if local:
            raise mlrun.errors.MLRunPreconditionFailedError(
                "Launch of local run inside the server is not allowed"
            )

        self._auth_info = auth_info

    def launch(
        self,
        runtime: mlrun.runtimes.BaseRuntime,
        task: Optional[
            Union["mlrun.run.RunTemplate", "mlrun.run.RunObject", dict]
        ] = None,
        handler: Optional[str] = None,
        name: Optional[str] = "",
        project: Optional[str] = "",
        params: Optional[dict] = None,
        inputs: Optional[dict[str, str | dict | list]] = None,
        out_path: Optional[str] = "",
        workdir: Optional[str] = "",
        artifact_path: Optional[str] = "",
        output_path: Optional[str] = "",
        watch: Optional[bool] = True,
        schedule: Optional[
            Union[str, mlrun.common.schemas.schedule.ScheduleCronTrigger]
        ] = None,
        hyperparams: Optional[dict[str, list]] = None,
        hyper_param_options: Optional[mlrun.model.HyperParamOptions] = None,
        verbose: Optional[bool] = None,
        scrape_metrics: Optional[bool] = None,
        local_code_path: Optional[str] = None,
        auto_build: Optional[bool] = None,
        param_file_secrets: Optional[dict[str, str]] = None,
        notifications: Optional[list[mlrun.model.Notification]] = None,
        returns: Optional[list[Union[str, dict[str, str]]]] = None,
        state_thresholds: Optional[dict[str, int]] = None,
        reset_on_run: Optional[bool] = None,
        retry: Optional[Union[mlrun.model.Retry, dict]] = None,
    ) -> mlrun.run.RunObject:
        self.enrich_runtime(runtime, project)

        run = self._create_run_object(task)

        run = self._enrich_run(
            runtime,
            run=run,
            handler=handler,
            project_name=project,
            name=name,
            params=params,
            inputs=inputs,
            returns=returns,
            hyperparams=hyperparams,
            hyper_param_options=hyper_param_options,
            verbose=verbose,
            scrape_metrics=scrape_metrics,
            output_path=output_path,
            workdir=workdir,
            notifications=notifications,
            state_thresholds=state_thresholds,
            retry=retry,
        )
        self._validate_run(runtime, run)

        if runtime.verbose:
            mlrun.utils.logger.info(f"Run:\n{run.to_yaml()}")

        if not runtime.is_child:
            mlrun.utils.logger.info(
                "Storing function",
                name=run.metadata.name,
                uid=run.metadata.uid,
            )
            self._store_function(runtime, run)

        execution = mlrun.execution.MLClientCtx.from_dict(
            run.to_dict(),
            runtime._get_db(),
            autocommit=False,
            is_api=True,
            store_run=False,
        )

        # create task generator (for child runs) from spec
        task_generator = mlrun.runtimes.generators.get_generator(
            run.spec, execution, param_file_secrets=param_file_secrets
        )
        if task_generator:
            # verify valid task parameters
            tasks = task_generator.generate(run)
            for task in tasks:
                self._validate_run_params(task.spec.parameters)

        # post verifications, store execution in db and run pre run hooks
        execution.store_run()
        self._configure_attempt_for_logging(run)
        runtime._pre_run(run, execution)  # hook for runtime specific prep

        last_err = None
        # If the runtime is nested, it means the hyper-run will run within a single instance of the run.
        # So while in the API, we consider the hyper-run as a single run, and then in the runtime itself when the
        # runtime is now a local runtime and therefore `self._is_nested == False`, we run each task as a separate run by
        # using the task generator
        if task_generator and not runtime._is_nested:
            # multiple runs (based on hyper params or params file)
            runner = runtime._run_many
            if hasattr(runtime, "_parallel_run_many") and task_generator.use_parallel():
                runner = runtime._parallel_run_many
            results = runner(task_generator, execution, run)
            mlrun.runtimes.utils.results_to_iter(results, run, execution)
            result = execution.to_dict()
            result = runtime._update_run_state(result, task=run)

        else:
            # single run
            try:
                # Skip retried run if it was aborted or deleted
                if self._should_skip_run(run):
                    run.status.state = mlrun.common.runtimes.constants.RunStates.aborted

                else:
                    runtime_handler = services.api.runtime_handlers.get_runtime_handler(
                        runtime.kind
                    )
                    runtime_handler.run(runtime, run, execution, self._auth_info)
            except mlrun.runtimes.utils.RunError as err:
                last_err = err

            finally:
                result = runtime._update_run_state(
                    task=run,
                    err=last_err,
                    run_format=mlrun.common.formatters.RunFormat.standard,
                )

        self._save_notifications(run)

        runtime._post_run(result, execution)  # hook for runtime specific cleanup

        return self._wrap_run_result(runtime, result, run, err=last_err)

    def enrich_runtime(
        self,
        runtime: "mlrun.runtimes.base.BaseRuntime",
        project_name: Optional[str] = "",
        full: bool = True,
        client_version: str = "",
    ):
        """
        Enrich the runtime object with the project spec and metadata.
        This is done only on the server side, since it's the source of truth for the project, and we want to keep the
        client side enrichment as minimal as possible.
        :param runtime:         the runtime object to enrich
        :param project_name:    the project name of the project to enrich the runtime with
        :param full:            whether to enrich the runtime with the project's full spec (before run)
                                e.g. mount, service account, etc.
        :param client_version:  MLRun client version
        """

        # if auth given in request ensure the function pod will have these auth env vars set, otherwise the job won't
        # be able to communicate with the api
        framework.api.utils.ensure_function_has_auth_set(
            runtime, self._auth_info, allow_empty_access_key=not full
        )

        if full:
            self._enrich_full_spec(runtime)
        # mask sensitive data after full spec enrichment in case auth was enriched by auto mount
        framework.api.utils.mask_function_sensitive_data(runtime, self._auth_info)

        # ensure the runtime has a project before we enrich it with the project's spec
        runtime.metadata.project = project_name or runtime.metadata.project
        if not runtime.metadata.project:
            raise mlrun.errors.MLRunMissingProjectError("Runtime must have a project")
        project = runtime._get_db().get_project(runtime.metadata.project)
        # this is mainly for tests with nop db
        # in normal use cases if no project is found we will get an error
        if project:
            if not isinstance(project, mlrun.projects.project.MlrunProject):
                project = mlrun.projects.project.MlrunProject.from_dict(project.dict())
            # there is no need to auto mount here as it was already done in the full spec enrichment with the auth info
            mlrun.projects.pipelines.enrich_function_object(
                project, runtime, copy_function=False, try_auto_mount=False
            )

        if (
            not runtime.spec.image
            and not runtime.requires_build()
            and runtime.kind in mlrun.mlconf.function_defaults.image_by_kind.to_dict()
            and not runtime.skip_image_enrichment()
        ):
            runtime.spec.image = mlrun.mlconf.function_defaults.image_by_kind.to_dict()[
                runtime.kind
            ]

        serving_spec = getattr(runtime, "serving_spec", None)
        if serving_spec and isinstance(runtime, KubejobRuntime | RemoteRuntime):
            serving_spec_volume = self._configure_serving_spec(
                client_version=client_version,
                function=runtime,
                project=project.name,
                serving_spec=serving_spec,
            )
            if serving_spec_volume:
                runtime.spec.volumes = runtime.spec.volumes + [
                    serving_spec_volume["volume"]
                ]
                runtime.spec.volume_mounts = runtime.spec.volume_mounts + [
                    serving_spec_volume["volumeMount"]
                ]

    def _enrich_run(
        self,
        runtime: "mlrun.runtimes.base.BaseRuntime",
        run: Union[RunSpec, RunTemplate],
        handler=None,
        project_name=None,
        name=None,
        params=None,
        inputs=None,
        returns=None,
        hyperparams=None,
        hyper_param_options=None,
        verbose=None,
        scrape_metrics=None,
        output_path=None,
        workdir=None,
        notifications: Optional[list[mlrun.model.Notification]] = None,
        state_thresholds: Optional[dict[str, int]] = None,
        retry: Optional[Union[mlrun.model.Retry, dict]] = None,
    ):
        run = super()._enrich_run(
            runtime=runtime,
            run=run,
            handler=handler,
            project_name=project_name,
            name=name,
            params=params,
            inputs=inputs,
            returns=returns,
            hyperparams=hyperparams,
            hyper_param_options=hyper_param_options,
            verbose=verbose,
            scrape_metrics=scrape_metrics,
            output_path=output_path,
            workdir=workdir,
            notifications=notifications,
            state_thresholds=state_thresholds,
            retry=retry,
        )

        self._handle_retry(run)
        run = self._pre_run_image_pull_secret_enrichment(run)
        self.enrich_and_validate_auth_token_name(run)
        return self._pre_run_scheduling_constraints_enrichment(runtime, run)

    @staticmethod
    def _handle_retry(run: mlrun.run.RunObject):
        if run.status.state != mlrun.common.runtimes.constants.RunStates.pending_retry:
            return

        run.status.state = mlrun.common.runtimes.constants.RunStates.running
        # retry_count may be None on first run attempt
        retry_count = run.status.retry_count or 0
        start_time = run.status.start_time

        # record retry metadata
        run.status.retries = run.status.retries or []
        run.status.retries.append(
            {
                "attempt": retry_count,
                "start_time": start_time,
                "end_time": run.status.end_time,
                "error": run.status.error,
            }
        )

        run.status.retry_count = retry_count + 1
        run.status.start_time = None
        # The combination of retry attempt label and requested logs `False` is required for the log collector to
        # collect logs from the current run attempt.
        run.metadata.labels[mlrun.common.constants.MLRunInternalLabels.retry] = str(
            run.status.retry_count
        )

    @staticmethod
    def _configure_attempt_for_logging(run: mlrun.run.RunObject):
        if not run.status.retry_count:
            # Run is not a retry, continue
            return

        framework.db.session.run_function_with_new_db_session(
            framework.utils.singletons.db.get_db().update_runs_requested_logs,
            uids=[run.metadata.uid],
            requested_logs=False,
        )

    def _pre_run_scheduling_constraints_enrichment(
        self,
        runtime: "mlrun.runtimes.base.BaseRuntime",
        run: mlrun.run.RunObject,
    ):
        """
        Enrich the run object with node selector, tolerations, and affinity before execution.

        Enrich the run object with the project's default node selector.
        This ensures the node selector is correctly set on the run
        while maintaining the runtime's integrity from system-specific project settings.

        Then, we apply preemption mode enrichment (if defined on the function).
        Preemption mode takes precedence over user-defined values,  and may modify or remove the node_selector,
        affinity, and tolerations fields to enforce scheduling behavior on preemptible/non-preemptible nodes.

        This ensures the pod will reflect the correct intent based on both user config and system scheduling policies.
        """
        # Start with function-level selector
        run.spec.node_selector = deepcopy(getattr(runtime.spec, "node_selector", {}))

        # Apply project-level enrichment if available
        if runtime._get_db():
            project = runtime._get_db().get_project(run.metadata.project)
            if project:
                project_node_selector = project.spec.default_function_node_selector
                resolved_node_selectors = mlrun.runtimes.utils.resolve_node_selectors(
                    project_node_selector, run.spec.node_selector
                )
                mlrun.k8s_utils.validate_node_selectors(resolved_node_selectors)
                run.spec.node_selector = resolved_node_selectors
        self._enrich_run_with_preemption_mode(runtime, run)
        return run

    def _enrich_run_with_preemption_mode(
        self,
        runtime: "mlrun.runtimes.base.BaseRuntime",
        run: mlrun.run.RunObject,
    ):
        """
        Apply preemption mode logic to node selector / affinity / tolerations on the run.
        """
        preemption_mode = getattr(runtime.spec, "preemption_mode", None)
        if not preemption_mode:
            return

        node_selector, tolerations, affinity = mlrun.k8s_utils.enrich_preemption_mode(
            preemption_mode,
            getattr(run.spec, "node_selector", None),
            getattr(runtime.spec, "tolerations", None),
            getattr(runtime.spec, "affinity", None),
        )

        tolerations, affinity = mlrun.k8s_utils.sanitize_scheduling_configuration(
            tolerations, affinity
        )
        self._set_run_spec_with_enriched_params(
            run,
            node_selector=node_selector,
            tolerations=tolerations,
            affinity=affinity,
        )

    def _set_run_spec_with_enriched_params(self, run, **fields):
        for key, value in fields.items():
            setattr(run.spec, key, value)

    def _pre_run_image_pull_secret_enrichment(self, run: Union[RunSpec, RunTemplate]):
        """
        Enrich the run object with the project's image pull secret.
        This ensures the image pull secret is correctly set on the run,
        either from the run spec or from the MLRun config
        """
        existing_image_pull_secret = getattr(run.spec, "image_pull_secret", None)
        run.spec.image_pull_secret = (
            existing_image_pull_secret
            or mlrun.config.config.function.spec.image_pull_secret.default
        )
        return run

    @staticmethod
    def _configure_serving_spec(
        client_version,
        function,
        project: str,
        serving_spec,
    ):
        serving_spec_volume = None
        if serving_spec is not None:
            # since environment variables have a limited size,
            # large serving specs are stored in config maps that are mounted to the pod
            serving_spec_len = len(serving_spec.encode("utf-8"))
            if serving_spec_len >= mlrun.mlconf.httpdb.nuclio.serving_spec_env_cutoff:
                if serving_spec_len >= SERVING_SPEC_MAX_LENGTH:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"The serving spec length exceeds the limit of {SERVING_SPEC_MAX_LENGTH}."
                    )
                # Compress and encode the serving spec
                compressed_serving_spec = gzip.compress(serving_spec.encode("utf-8"))
                encoded_serving_spec = base64.b64encode(compressed_serving_spec).decode(
                    "utf-8"
                )

                function_name = mlrun.runtimes.nuclio.function.get_fullname(
                    function.metadata.name, project, function.metadata.tag
                )
                k8s_helper = framework.utils.singletons.k8s.get_k8s_helper()
                confmap_name = k8s_helper.ensure_configmap(
                    mlrun.common.constants.MLRUN_SERVING_CONF,
                    function_name,
                    {
                        mlrun.common.constants.MLRUN_SERVING_SPEC_FILENAME: encoded_serving_spec
                    },
                    labels={mlrun_constants.MLRunInternalLabels.created: "true"},
                    project=project,
                )
                volume_name = mlrun.common.constants.MLRUN_SERVING_CONF
                volume_mount = {
                    "name": volume_name,
                    "mountPath": mlrun.common.constants.MLRUN_SERVING_SPEC_MOUNT_PATH,
                    "readOnly": True,
                }

                serving_spec_volume = {
                    "volume": {
                        "name": volume_name,
                        "configMap": {"name": confmap_name},
                    },
                    "volumeMount": volume_mount,
                }
            else:
                function.spec.env["SERVING_SPEC_ENV"] = serving_spec
        return serving_spec_volume

    @staticmethod
    def _should_skip_run(run: mlrun.run.RunObject) -> bool:
        """
        Determine whether a retried run should be skipped based on its state.
        A run should be skipped if it is in 'pending_retry' state and was either aborted or deleted after being
        scheduled for retry.
        """
        if run.status.state != mlrun.common.runtimes.constants.RunStates.pending_retry:
            return False

        # fetch the run from the db to check if it was deleted after the retry attempt
        db = framework.utils.singletons.db.get_db()
        try:
            db_run = framework.db.session.run_function_with_new_db_session(
                db.read_run,
                uid=run.metadata.uid,
                project=run.metadata.project,
            )
        except mlrun.errors.MLRunNotFoundError:
            mlrun.utils.logger.info(
                "Skipping retry for run - run was deleted",
                uid=run.metadata.uid,
                project=run.metadata.project,
            )
            return True

        # check if it was aborted after the retry attempt
        if (
            db_run.get("status", {}).get("state")
            == mlrun.common.runtimes.constants.RunStates.aborted
        ):
            mlrun.utils.logger.info(
                "Skipping retry for run - run was aborted",
                uid=run.metadata.uid,
                project=run.metadata.project,
            )
            return True

        return False

    def _enrich_full_spec(
        self,
        runtime: "mlrun.runtimes.base.BaseRuntime",
    ):
        # If this was triggered by the UI, we will need to attempt auto-mount based on auto-mount
        # config and params passed in the auth_info.
        # If this was triggered by the SDK, then auto-mount was already attempted and will be skipped.
        framework.api.utils.try_perform_auto_mount(runtime, self._auth_info)

        # Validate function's service-account, based on allowed SAs for the project,
        # if existing in a project-secret.
        framework.api.utils.process_function_service_account(runtime, self._auth_info)

        framework.api.utils.ensure_function_security_context(runtime, self._auth_info)

        existing_image_pull_secret = runtime.spec.image_pull_secret
        runtime.spec.image_pull_secret = (
            existing_image_pull_secret
            or mlrun.config.config.function.spec.image_pull_secret.default
        )

    def _save_notifications(self, runobj):
        if not self._run_has_valid_notifications(runobj):
            return

        # If in the api server, we can assume that watch=False, so we save notification
        # configs to the DB, for the run monitor to later pick up and push.
        framework.db.session.run_function_with_new_db_session(
            services.api.crud.Notifications().store_run_notifications,
            runobj.spec.notifications,
            runobj.metadata.uid,
            runobj.metadata.project,
        )

    def _store_function(
        self, runtime: mlrun.runtimes.base.BaseRuntime, run: mlrun.run.RunObject
    ):
        run.metadata.labels[mlrun_constants.MLRunInternalLabels.kind] = runtime.kind

        # Server-side owner enrichment: override client-provided owner with authenticated username.
        # In authenticated environments (e.g., IG4), auth_info.username is the source of truth.
        # This ensures the owner label reflects the authenticated user rather than the local user
        # on the client machine (e.g., 'jovyan' in Jupyter notebooks).
        # For CE/unauthenticated deployments, auth_info.username will be None, preserving
        # any existing owner label from client-side enrichment.
        if self._auth_info and self._auth_info.username:
            run.metadata.labels[mlrun_constants.MLRunInternalLabels.owner] = (
                self._auth_info.username
            )

        # Replace {{run.user}} template in output_path with the final owner value.
        # This must happen after owner enrichment to ensure correct substitution.
        run.spec.output_path = mlrun.runtimes.utils.resolve_run_user_template(
            run.spec.output_path,
            run.metadata.labels.get(mlrun_constants.MLRunInternalLabels.owner),
        )

        db = runtime._get_db()
        if db and runtime.kind != "handler":
            struct = runtime.to_dict()
            hash_key = db.store_function(
                struct, runtime.metadata.name, runtime.metadata.project, versioned=True
            )
            run.spec.function = runtime._function_uri(hash_key=hash_key)

    def _validate_run(
        self,
        runtime: "mlrun.runtimes.BaseRuntime",
        run: "mlrun.run.RunObject",
    ):
        if (
            mlrun.runtimes.RuntimeKinds.is_local_runtime(runtime.kind)
            and not mlrun.mlconf.httpdb.jobs.allow_local_run
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Local runtimes can not be run through API (not locally)"
            )

        self._validate_state_thresholds(run.spec.state_thresholds)
        self._validate_retry(runtime.kind, run.spec.retry)

        if (
            mlrun.runtimes.RuntimeKinds.requires_image_name_for_execution(runtime.kind)
            and not runtime.spec.image
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"This runtime kind ({runtime.kind}) must have a valid image"
            )

        super()._validate_run(runtime, run)

    @staticmethod
    def _validate_state_thresholds(
        state_thresholds: Optional[dict[str, str]] = None,
    ):
        """
        Validate the state thresholds
        If threshold is:
            - None - will use default
            - -1 - infinity
            - otherwise - validate it's a valid time string
        """
        if state_thresholds is None:
            return

        for state, threshold in state_thresholds.items():
            if state not in mlrun.common.runtimes.constants.ThresholdStates.all():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Invalid state {state} for state threshold, must be one of "
                    f"{mlrun.common.runtimes.constants.ThresholdStates.all()}"
                )

            if threshold is None:
                continue

            if not isinstance(threshold, str):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Threshold '{threshold}' for state '{state}' must be a string"
                )

            try:
                framework.utils.helpers.time_string_to_seconds(threshold)
            except Exception as exc:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Threshold '{threshold}' for state '{state}' is not a valid timelength string. "
                    f"Error: {mlrun.errors.err_to_str(exc)}"
                ) from exc

    @staticmethod
    def _validate_retry(runtime_kind: str, retry: Optional["mlrun.model.Retry"]):
        if retry is None or not retry.count:
            return

        if retry.count is not None and retry.count < 0:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Retry count must be at least 0, got {retry.count}"
            )

        if runtime_kind not in mlrun.runtimes.RuntimeKinds.retriable_runtimes():
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Retry is not supported for runtime kind {runtime_kind}, supported kinds are: "
                f"{mlrun.runtimes.RuntimeKinds.retriable_runtimes()}"
            )

        backoff = retry.backoff
        if backoff is not None and backoff.base_delay is not None:
            min_base_delay = mlrun.mlconf.function.spec.retry.backoff.min_base_delay
            try:
                base_delay_seconds = framework.utils.helpers.time_string_to_seconds(
                    backoff.base_delay,
                    mlrun.mlconf.function.spec.retry.backoff.min_base_delay,
                )
            except ValueError as exc:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Retry backoff base_delay must be at least {min_base_delay}, got {backoff.base_delay}"
                ) from exc

            staleness_threshold_seconds = mlrun.mlconf.get_run_retry_staleness_threshold_timedelta().total_seconds()
            staleness_threshold_seconds = int(staleness_threshold_seconds)
            max_delay = int(base_delay_seconds * retry.count)
            if max_delay >= staleness_threshold_seconds:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Retry backoff base_delay {backoff.base_delay} * retry count {retry.count} "
                    f"must be less than {staleness_threshold_seconds} seconds, got {max_delay} seconds"
                )

    def enrich_and_validate_auth_token_name(
        self, object: Union[mlrun.run.RunObject, mlrun.runtimes.RemoteRuntime]
    ):
        if not (mlrun.mlconf.is_iguazio_v4_mode()):
            return

        # Get the provided token name, if any
        provided_token_name = (object.spec.auth or {}).get("token_name")

        # Use the token resolution logic that validates existence and expiration
        token_name = services.api.utils.helpers.resolve_auth_token_name(
            user_id=self._auth_info.user_id, provided_token_name=provided_token_name
        )

        mlrun.utils.helpers.set_auth_token_name(object.spec, token_name)


# Once this file is imported it will set the container server side launcher
@containers.override(mlrun.launcher.factory.LauncherContainer)
class ServerSideLauncherContainer(containers.DeclarativeContainer):
    server_side_launcher = providers.Factory(ServerSideLauncher)
