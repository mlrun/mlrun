from typing import Callable, Optional, Union

from sqlalchemy.orm import Session

import mlrun.api.schemas
import mlrun.utils.singleton
from mlrun.api.api.utils import (
    apply_enrichment_and_validation_on_function,
    get_run_db_instance,
    get_scheduler,
)


class Workflows(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def create_function(
        self,
        run_name: str,
        project: str,
        db_session: Session,
        auth_info: mlrun.api.schemas.AuthInfo,
        access_key: str = None,
        **kwargs,
    ):
        _function = mlrun.new_function(name=run_name, project=project, **kwargs)

        run_db = get_run_db_instance(db_session)
        _function.set_db_connection(run_db)

        if access_key:
            _function.metadata.credentials.access_key = access_key

        apply_enrichment_and_validation_on_function(
            function=_function,
            auth_info=auth_info,
        )

        _function.save()
        return _function

    def execute_function(
        self,
        function: mlrun.runtimes.BaseRuntime,
        project,
        workflow_spec,
        artifact_path,
        namespace,
        workflow_name=None,
        db_session: Session = None,
        auth_info: mlrun.api.schemas.AuthInfo = None,
        **kwargs,
    ):
        runspec_kwargs = kwargs.get("runspec_kwargs", {})
        runspec = _create_run_object_for_workflow_runner(
            project=project,
            workflow_spec=workflow_spec,
            artifact_path=artifact_path,
            namespace=namespace,
            workflow_name=workflow_name,
            workflow_handler=workflow_spec.handler,
            **runspec_kwargs,
        )

        if workflow_spec.schedule:
            # Creating scheduled object:
            scheduled_object = {
                "task": runspec.to_dict(),
                "schedule": workflow_spec.schedule,
            }
            # Creating schedule:
            get_scheduler().create_schedule(
                db_session=db_session,
                auth_info=auth_info,
                project=project.metadata.name,
                name=function.metadata.name,
                kind=mlrun.api.schemas.ScheduleKinds.job,
                scheduled_object=scheduled_object,
                cron_trigger=workflow_spec.schedule,
                labels=function.metadata.labels,
            )
        else:
            return function.run(runspec=runspec, artifact_path=artifact_path, **kwargs)


def _create_run_object_for_workflow_runner(
    project,
    workflow_spec,
    artifact_path: Optional[str] = None,
    namespace: Optional[str] = None,
    workflow_name: Optional[str] = None,
    workflow_handler: Union[str, Callable] = None,
    **kwargs,
) -> mlrun.RunObject:
    """
    Creating run object for the load_and_run function.

    :param project:             project object that matches the workflow
    :param workflow_spec:       spec of the workflow to run
    :param artifact_path:       artifact path target for the run
    :param namespace:           kubernetes namespace if other than default
    :param workflow_name:       name of the workflow to override the one in the workflow spec.
    :param workflow_handler:    handler of the workflow to override the one in the workflow spec.
    :param kwargs:              dictionary with "spec" and "metadata" keys with dictionaries as values that are
                                corresponding to the keys.

    :return:    a RunObject with the desired spec and metadata with labels.
    """
    spec_kwargs, metadata_kwargs = (
        kwargs.get("spec"),
        kwargs.get("metadata"),
    ) if kwargs else {}, {}
    spec = {
        "parameters": {
            "url": project.spec.source,
            "project_name": project.metadata.name,
            "workflow_name": workflow_name or workflow_spec.name,
            "workflow_path": workflow_spec.path,
            "workflow_arguments": workflow_spec.args,
            "artifact_path": artifact_path,
            "workflow_handler": workflow_handler or workflow_spec.handler,
            "namespace": namespace,
            "ttl": workflow_spec.ttl,
            "engine": workflow_spec.engine,
            "local": workflow_spec.run_local,
        },
        "handler": "mlrun.projects.load_and_run",
    }
    metadata = {"name": workflow_name}
    spec.update(spec_kwargs)
    metadata.update(metadata_kwargs)

    # Creating object:
    run_object = mlrun.RunObject.from_dict({"spec": spec, "metadata": metadata})

    # Setting labels:
    return run_object.set_label("job-type", "workflow-runner").set_label(
        "workflow", workflow_name or workflow_spec.name
    )
