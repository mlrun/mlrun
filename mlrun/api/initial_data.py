import mlrun.api.db.sqldb.db
import sqlalchemy.orm
import mlrun.artifacts
import mlrun.artifacts.dataset
from mlrun.api.db.init_db import init_db
from mlrun.api.db.session import create_session, close_session
from mlrun.utils import logger


def init_data() -> None:
    logger.info("Creating initial data")
    db_session = create_session()
    try:
        init_db(db_session)
        _perform_data_migrations(db_session)
    finally:
        close_session(db_session)
    logger.info("Initial data created")


def _perform_data_migrations(db_session: sqlalchemy.orm.Session):
    # FileDB is not really a thing anymore, so using SQLDB directly
    db = mlrun.api.db.sqldb.db.SQLDB("")
    logger.info("Performing data migrations")
    _fix_datasets_large_previews(db, db_session)


def _fix_datasets_large_previews(
    db: mlrun.api.db.sqldb.db.SQLDB, db_session: sqlalchemy.orm.Session
):
    # get all artifacts
    artifacts = db._find_artifacts(db_session, None, "*")
    for artifact in artifacts:
        try:
            artifact_dict = artifact.struct
            if (
                artifact_dict
                and artifact_dict.get("kind") == mlrun.artifacts.DatasetArtifact.kind
            ):
                header = artifact_dict.get("header", [])
                if header and len(header) > mlrun.artifacts.dataset.max_preview_columns:
                    logger.debug(
                        "Found dataset artifact with more than allowed columns in preview fields. Fixing",
                        artifact=artifact_dict,
                    )
                    columns_to_remove = header[
                        mlrun.artifacts.dataset.max_preview_columns :
                    ]

                    # align preview
                    if artifact_dict.get("preview"):
                        new_preview = []
                        for preview_row in artifact_dict["preview"]:
                            # sanity
                            if (
                                len(preview_row)
                                < mlrun.artifacts.dataset.max_preview_columns
                            ):
                                logger.warning(
                                    "Found artifact with more than allowed columns in header definition, "
                                    "but preview data is valid. Leaving preview as is",
                                    artifact=artifact_dict,
                                )
                            new_preview.append(
                                preview_row[
                                    : mlrun.artifacts.dataset.max_preview_columns
                                ]
                            )

                        artifact_dict["preview"] = new_preview

                    # align stats
                    for column_to_remove in columns_to_remove:
                        if column_to_remove in artifact_dict.get("stats", {}):
                            del artifact_dict["stats"][column_to_remove]

                    # align schema
                    if artifact_dict.get("schema", {}).get("fields"):
                        new_schema_fields = []
                        for field in artifact_dict["schema"]["fields"]:
                            if field.get("name") not in columns_to_remove:
                                new_schema_fields.append(field)
                        artifact_dict["schema"]["fields"] = new_schema_fields

                    # lastly, align headers
                    artifact_dict["header"] = header[
                        : mlrun.artifacts.dataset.max_preview_columns
                    ]
                    logger.debug(
                        "Fixed dataset artifact preview fields. Storing",
                        artifact=artifact_dict,
                    )
                    db._store_artifact(
                        db_session,
                        artifact.key,
                        artifact_dict,
                        artifact.uid,
                        project=artifact.project,
                        tag_artifact=False,
                    )
        except Exception as exc:
            logger.warning(
                "Failed fixing dataset artifact large preview. Continuing", exc=exc,
            )


def main() -> None:
    init_data()


if __name__ == "__main__":
    main()
