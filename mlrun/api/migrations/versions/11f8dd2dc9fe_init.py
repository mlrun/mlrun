"""init

Revision ID: 11f8dd2dc9fe
Revises:
Create Date: 2020-10-06 15:50:35.588592

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "11f8dd2dc9fe"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(255), nullable=True),
        sa.Column("project", sa.String(255), nullable=True),
        sa.Column("uid", sa.String(255), nullable=True),
        sa.Column("updated", sa.TIMESTAMP(), nullable=True),
        sa.Column("body", sa.BLOB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("uid", "project", "key", name="_artifacts_uc"),
    )
    op.create_table(
        "functions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("project", sa.String(255), nullable=True),
        sa.Column("uid", sa.String(255), nullable=True),
        sa.Column("body", sa.BLOB(), nullable=True),
        sa.Column("updated", sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "project", "uid", name="_functions_uc"),
    )
    op.create_table(
        "logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("uid", sa.String(255), nullable=True),
        sa.Column("project", sa.String(255), nullable=True),
        sa.Column("body", sa.BLOB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("description", sa.String(255), nullable=True),
        sa.Column("owner", sa.String(255), nullable=True),
        sa.Column("source", sa.String(255), nullable=True),
        sa.Column("spec", sa.BLOB(), nullable=True),
        sa.Column("created", sa.TIMESTAMP(), nullable=True),
        sa.Column("state", sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="_projects_uc"),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("uid", sa.String(255), nullable=True),
        sa.Column("project", sa.String(255), nullable=True),
        sa.Column("iteration", sa.Integer(), nullable=True),
        sa.Column("state", sa.String(255), nullable=True),
        sa.Column("body", sa.BLOB(), nullable=True),
        sa.Column("start_time", sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("uid", "project", "iteration", name="_runs_uc"),
    )
    op.create_table(
        "schedules_v2",
        sa.Column("project", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("kind", sa.String(255), nullable=True),
        sa.Column("desired_state", sa.String(255), nullable=True),
        sa.Column("state", sa.String(255), nullable=True),
        sa.Column("creation_time", sa.TIMESTAMP(), nullable=True),
        sa.Column("cron_trigger_str", sa.String(255), nullable=True),
        sa.Column("struct", sa.BLOB(), nullable=True),
        sa.PrimaryKeyConstraint("project", "name"),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="_users_uc"),
    )
    op.create_table(
        "artifacts_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("value", sa.String(255), nullable=True),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["parent"], ["artifacts.id"],),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "parent", name="_artifacts_labels_uc"),
    )
    op.create_table(
        "artifacts_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project", sa.String(255), nullable=True),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["obj_id"], ["artifacts.id"],),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project", "name", "obj_id", name="_artifacts_tags_uc"),
    )
    op.create_table(
        "functions_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("value", sa.String(255), nullable=True),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["parent"], ["functions.id"],),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "parent", name="_functions_labels_uc"),
    )
    op.create_table(
        "functions_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project", sa.String(255), nullable=True),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.Column("obj_name", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["obj_id"], ["functions.id"],),
        sa.ForeignKeyConstraint(["obj_name"], ["functions.name"],),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project", "name", "obj_name", name="_functions_tags_uc"),
    )
    op.create_table(
        "project_users",
        sa.Column("project_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"],),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"],),
    )
    op.create_table(
        "runs_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("value", sa.String(255), nullable=True),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["parent"], ["runs.id"],),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "parent", name="_runs_labels_uc"),
    )
    op.create_table(
        "runs_tags",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project", sa.String(255), nullable=True),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("obj_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["obj_id"], ["runs.id"],),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project", "name", "obj_id", name="_runs_tags_uc"),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("runs_tags")
    op.drop_table("runs_labels")
    op.drop_table("project_users")
    op.drop_table("functions_tags")
    op.drop_table("functions_labels")
    op.drop_table("artifacts_tags")
    op.drop_table("artifacts_labels")
    op.drop_table("users")
    op.drop_table("schedules_v2")
    op.drop_table("runs")
    op.drop_table("projects")
    op.drop_table("logs")
    op.drop_table("functions")
    op.drop_table("artifacts")
    # ### end Alembic commands ###
