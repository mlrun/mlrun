"""create new artifact table

Revision ID: 678bbf601122
Revises: 6401142f2d7c
Create Date: 2023-01-10 12:23:49.431054

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "678bbf601122"
down_revision = "6401142f2d7c"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.rename_table("artifacts", "legacy_artifacts")
    op.rename_table("artifacts_labels", "legacy_artifacts_labels")
    op.rename_table("artifacts_tags", "legacy_artifacts_tags")

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("iter", sa.Integer(), nullable=True),
        sa.Column("project", sa.String(length=255), nullable=True),
        sa.Column("uid", sa.String(length=255), nullable=True),
        sa.Column("producer_uid", sa.String(length=255), nullable=True),
        sa.Column("created", sa.TIMESTAMP(), nullable=True),
        sa.Column("updated", sa.TIMESTAMP(), nullable=True),
        sa.Column("object", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("uid", "project", "name", name="_artifacts_uc"),
    )
    op.create_table(
        "artifacts_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("value", sa.String(length=255), nullable=True),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"],
            ["artifacts_new.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "parent", name="_artifacts_labels_uc"),
    )
    op.create_table(
        "artifacts_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project", sa.String(length=255), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.Column("obj_name", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["obj_id"],
            ["artifacts.id"],
        ),
        sa.ForeignKeyConstraint(
            ["obj_name"],
            ["artifacts.name"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project", "name", "obj_name", name="_artifacts_tags_uc"),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("artifacts_tags")
    op.drop_table("artifacts_labels")
    op.drop_table("artifacts")

    op.rename_table("legacy_artifacts", "artifacts")
    op.rename_table("legacy_artifacts_labels", "artifacts_labels")
    op.rename_table("legacy_artifacts_tags", "artifacts_tags")

    # ### end Alembic commands ###
