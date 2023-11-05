"""add index migration

Revision ID: 59061f6e2a87
Revises: 27ed4ecb734c
Create Date: 2023-11-05 12:43:53.787957

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "59061f6e2a87"
down_revision = "27ed4ecb734c"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index("idx_runs_project_uid", "runs", ["id", "project"], unique=False)

    op.create_index(
        "idx_artifacts_labels_name_value",
        "artifacts_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_entities_labels_name_value",
        "entities_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_feature_sets_labels_name_value",
        "feature_sets_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_projects_labels_name_value",
        "projects_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_functions_labels_name_value",
        "functions_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_features_labels_name_value",
        "features_labels",
        ["name", "value"],
        unique=False,
    )
    op.create_index(
        "idx_feature_vectors_labels_name_value",
        "feature_vectors_labels",
        ["name", "value"],
        unique=False,
    )

    op.create_index(
        "idx_runs_labels_name_value", "runs_labels", ["name", "value"], unique=False
    )

    op.create_index(
        "idx_schedules_v2_labels_name_value",
        "schedules_v2_labels",
        ["name", "value"],
        unique=False,
    )
    # ### end Alembic commands ###
