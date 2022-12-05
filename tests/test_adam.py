import os

import mlrun
import mlrun.config
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.k8s_utils
import mlrun.api.db.session
import mlrun.api.db.sqldb.session
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.project_member
import mlrun.api.initial_data

# db_host_internal = "host.docker.internal"



def test_notification(monkeypatch):
    db_host_external = "localhost"
    db_user = "root"
    db_port = 3306
    db_name = "mlrun"
    db_dsn = f"mysql+pymysql://{db_user}@{db_host_external}:{db_port}/{db_name}"
    mlrun.config.config.dbpath = db_dsn
    mlrun.config.config.httpdb.db_type = "sqldb"
    mlrun.config.config.httpdb.dsn = db_dsn
    mlrun.config.config.namespace = "default"
    mlrun.config.config.httpdb.db.backup.mode = "disabled"
    os.environ["MLRUN_HTTPDB__DSN"] = db_dsn

    mlrun.api.db.sqldb.session._session_maker = None

    mlrun.api.db.sqldb.session._init_engine(db_dsn)
    mlrun.api.initial_data.init_data(perform_migrations_if_needed=True)
    mlrun.api.utils.singletons.db.initialize_db()
    mlrun.api.utils.singletons.project_member.initialize_project_member()
    db_session = mlrun.api.db.session.create_session()

    project = mlrun.new_project("test", save=False)
    project_schema = mlrun.api.schemas.Project(**project.to_dict())
    mlrun.api.crud.Projects().create_project(db_session, project_schema)

    monkeypatch.setattr(
        mlrun.k8s_utils,
        "get_k8s_helper",
        lambda *args, **kwargs: mlrun.k8s_utils.K8sHelper(
            mlrun.config.config.namespace, config_file="~/.kube/config"
        ),
    )

    function = mlrun.new_function(
        "function-from-module",
        kind="job",
        project="test",
        image="mlrun/mlrun",
    )
    run = function.run(handler="json.dumps", params={"obj": {"x": 99}})
