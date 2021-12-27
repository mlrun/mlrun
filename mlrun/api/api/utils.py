import re
import traceback
import typing
from hashlib import sha1
from http import HTTPStatus
from os import environ
from pathlib import Path

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.errors
from mlrun.api import schemas
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.schemas import SecretProviderName
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.utils.singletons.logs_dir import get_logs_dir
from mlrun.api.utils.singletons.scheduler import get_scheduler
from mlrun.config import config
from mlrun.db.sqldb import SQLDB as SQLRunDB
from mlrun.run import import_function, new_function
from mlrun.runtimes.utils import enrich_function_from_dict
from mlrun.utils import get_in, logger, parse_versioned_object_uri


def log_and_raise(status=HTTPStatus.BAD_REQUEST.value, **kw):
    logger.error(str(kw))
    # TODO: 0.6.6 is the last version expecting the error details to be under reason, when it's no longer a relevant
    #  version can be changed to details=kw
    raise HTTPException(status_code=status, detail={"reason": kw})


def log_path(project, uid) -> Path:
    return project_logs_path(project) / uid


def project_logs_path(project) -> Path:
    return get_logs_dir() / project


def get_obj_path(schema, path, user=""):
    if path.startswith("/User/"):
        user = user or environ.get("V3IO_USERNAME", "admin")
        path = "v3io:///users/" + user + path[5:]
        schema = schema or "v3io"
    elif path.startswith("/v3io"):
        path = "v3io://" + path[len("/v3io") :]
        schema = schema or "v3io"
    elif config.httpdb.data_volume and path.startswith(config.httpdb.data_volume):
        data_volume_prefix = config.httpdb.data_volume
        if data_volume_prefix.endswith("/"):
            data_volume_prefix = data_volume_prefix[:-1]
        if config.httpdb.real_path:
            path_from_volume = path[len(data_volume_prefix) :]
            if path_from_volume.startswith("/"):
                path_from_volume = path_from_volume[1:]
            path = str(Path(config.httpdb.real_path) / Path(path_from_volume))
    if schema:
        schema_prefix = schema + "://"
        if not path.startswith(schema_prefix):
            return schema + "://" + path
    return path


def get_secrets(auth_info: mlrun.api.schemas.AuthInfo):
    return {
        "V3IO_ACCESS_KEY": auth_info.data_session,
    }


def get_run_db_instance(db_session: Session,):
    db = get_db()
    if isinstance(db, SQLDB):
        run_db = SQLRunDB(db.dsn, db_session)
    else:
        run_db = db.db
    run_db.connect()
    return run_db


def parse_submit_run_body(data):
    task = data.get("task")
    function_dict = data.get("function")
    function_url = data.get("functionUrl")
    if not function_url and task:
        function_url = get_in(task, "spec.function")
    if not (function_dict or function_url) or not task:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="bad JSON, need to include function/url and task objects",
        )
    return function_dict, function_url, task


def _generate_function_and_task_from_submit_run_body(
    db_session: Session, auth_info: mlrun.api.schemas.AuthInfo, data
):
    function_dict, function_url, task = parse_submit_run_body(data)
    # TODO: block exec for function["kind"] in ["", "local]  (must be a
    # remote/container runtime)

    if function_dict and not function_url:
        function = new_function(runtime=function_dict)
    else:
        if "://" in function_url:
            function = import_function(
                url=function_url, project=task.get("metadata", {}).get("project")
            )
        else:
            project, name, tag, hash_key = parse_versioned_object_uri(function_url)
            function_record = get_db().get_function(
                db_session, name, project, tag, hash_key
            )
            if not function_record:
                log_and_raise(
                    HTTPStatus.NOT_FOUND.value,
                    reason=f"runtime error: function {function_url} not found",
                )
            function = new_function(runtime=function_record)

        if function_dict:
            # The purpose of the function dict is to enable the user to override configurations of the existing function
            # without modifying it - to do that we're creating a function object from the request function dict and
            # assign values from it to the main function object
            function = enrich_function_from_dict(function, function_dict)

    # if auth given in request ensure the function pod will have these auth env vars set, otherwise the job won't
    # be able to communicate with the api
    ensure_function_has_auth_set(function, auth_info)

    # if this was triggered by the UI, we will need to attempt auto-mount based on auto-mount config and params passed
    # in the auth_info. If this was triggered by the SDK, then auto-mount was already attempted and will be skipped.
    try_perform_auto_mount(function, auth_info)

    # Validate function's service-account, based on allowed SAs for the project, if existing in a project-secret.
    process_function_service_account(function)
    return function, task


async def submit_run(db_session: Session, auth_info: mlrun.api.schemas.AuthInfo, data):
    _, _, _, response = await run_in_threadpool(
        _submit_run, db_session, auth_info, data
    )
    return response


def ensure_function_has_auth_set(function, auth_info: mlrun.api.schemas.AuthInfo):
    if (
        function.kind
        and function.kind not in mlrun.runtimes.RuntimeKinds.local_runtimes()
        and mlrun.api.utils.auth.verifier.AuthVerifier().is_jobs_auth_required()
    ):
        if (
            function.metadata.credentials.access_key
            == mlrun.model.Credentials.generate_access_key
        ):
            if not auth_info.access_key:
                auth_info.access_key = mlrun.api.utils.auth.verifier.AuthVerifier().get_or_create_access_key(
                    auth_info.session
                )
            function.metadata.credentials.access_key = auth_info.access_key
        if not function.metadata.credentials.access_key:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Function access key must be set (function.metadata.credentials.access_key)"
            )
        auth_env_vars = {
            "MLRUN_AUTH_SESSION": function.metadata.credentials.access_key,
        }
        for key, value in auth_env_vars.items():
            function.set_env(key, value)


def try_perform_auto_mount(function, auth_info: mlrun.api.schemas.AuthInfo):
    if (
        function.kind in mlrun.runtimes.RuntimeKinds.local_runtimes()
        or function.spec.disable_auto_mount
    ):
        return
    # Retrieve v3io auth params from the caller auth info
    override_params = {}
    if auth_info.data_session or auth_info.access_key:
        override_params["access_key"] = auth_info.data_session or auth_info.access_key
    if auth_info.username:
        override_params["user"] = auth_info.username

    function.try_auto_mount_based_on_config(override_params)


def process_function_service_account(function):
    allowed_service_accounts = mlrun.api.crud.secrets.Secrets().get_secret(
        function.metadata.project,
        SecretProviderName.kubernetes,
        mlrun.api.crud.secrets.Secrets().generate_service_account_secret_key("allowed"),
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
    )
    if allowed_service_accounts:
        allowed_service_accounts = [
            service_account.strip()
            for service_account in allowed_service_accounts.split(",")
        ]

    default_service_account = mlrun.api.crud.secrets.Secrets().get_secret(
        function.metadata.project,
        SecretProviderName.kubernetes,
        mlrun.api.crud.secrets.Secrets().generate_service_account_secret_key("default"),
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
    )

    # Sanity check on project configuration
    if (
        default_service_account
        and allowed_service_accounts
        and default_service_account not in allowed_service_accounts
    ):
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Default service account {default_service_account} is not in list of allowed "
            + f"service accounts {allowed_service_accounts}"
        )

    function.validate_and_enrich_service_account(
        allowed_service_accounts, default_service_account
    )


def _submit_run(
    db_session: Session, auth_info: mlrun.api.schemas.AuthInfo, data
) -> typing.Tuple[str, str, str, typing.Dict]:
    """
    :return: Tuple with:
        1. str of the project of the run
        2. str of the kind of the function of the run
        3. str of the uid of the run that started execution (None when it was scheduled)
        4. dict of the response info
    """
    run_uid = None
    project = None
    try:
        fn, task = _generate_function_and_task_from_submit_run_body(
            db_session, auth_info, data
        )
        if (
            not fn.kind
            or fn.kind in mlrun.runtimes.RuntimeKinds.local_runtimes()
            and not mlrun.mlconf.httpdb.jobs.allow_local_run
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Local runtimes can not be run through API (not locally)"
            )
        run_db = get_run_db_instance(db_session)
        fn.set_db_connection(run_db, True)
        logger.info("Submitting run", function=fn.to_dict(), task=task)
        # fn.spec.rundb = "http://mlrun-api:8080"
        schedule = data.get("schedule")
        if schedule:
            cron_trigger = schedule
            if isinstance(cron_trigger, dict):
                cron_trigger = schemas.ScheduleCronTrigger(**cron_trigger)
            schedule_labels = task["metadata"].get("labels")
            get_scheduler().create_schedule(
                db_session,
                auth_info,
                task["metadata"]["project"],
                task["metadata"]["name"],
                schemas.ScheduleKinds.job,
                data,
                cron_trigger,
                schedule_labels,
            )
            project = task["metadata"]["project"]

            response = {
                "schedule": schedule,
                "project": task["metadata"]["project"],
                "name": task["metadata"]["name"],
            }
        else:
            run = fn.run(task, watch=False)
            run_uid = run.metadata.uid
            project = run.metadata.project
            if run:
                response = run.to_dict()

    except HTTPException:
        logger.error(traceback.format_exc())
        raise
    except mlrun.errors.MLRunHTTPStatusError:
        raise
    except Exception as err:
        logger.error(traceback.format_exc())
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=f"runtime error: {err}")

    logger.info("Run submission succeeded", response=response)
    return project, fn.kind, run_uid, {"data": response}


# uid is hexdigest of sha1 value, which is double the digest size due to hex encoding
hash_len = sha1().digest_size * 2
uid_regex = re.compile(f"^[0-9a-f]{{{hash_len}}}$", re.IGNORECASE)


def parse_reference(reference: str):
    tag = None
    uid = None
    regex_match = uid_regex.match(reference)
    if not regex_match:
        tag = reference
    else:
        uid = regex_match.string
    return tag, uid
