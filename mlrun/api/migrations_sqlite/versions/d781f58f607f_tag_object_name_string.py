"""tag object name string

Revision ID: d781f58f607f
Revises: e1dd5983c06b
Create Date: 2021-07-29 16:06:45.555323

"""
import sqlalchemy as sa
from alembic import op

from mlrun.api.utils.db.sql_collation import SQLCollationUtil

# revision identifiers, used by Alembic.
revision = "d781f58f607f"
down_revision = "deac06871ace"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("feature_sets_tags") as batch_op:
        batch_op.alter_column(
            column_name="obj_name",
            type_=sa.String(255, collation=SQLCollationUtil.collation()),
        )
    with op.batch_alter_table("feature_vectors_tags") as batch_op:
        batch_op.alter_column(
            column_name="obj_name",
            type_=sa.String(255, collation=SQLCollationUtil.collation()),
        )
    with op.batch_alter_table("functions_tags") as batch_op:
        batch_op.alter_column(
            column_name="obj_name",
            type_=sa.String(255, collation=SQLCollationUtil.collation()),
        )


def downgrade():
    with op.batch_alter_table("functions_tags") as batch_op:
        batch_op.alter_column(column_name="obj_name", type_=sa.Integer())
    with op.batch_alter_table("feature_vectors_tags") as batch_op:
        batch_op.alter_column(column_name="obj_name", type_=sa.Integer())
    with op.batch_alter_table("feature_sets_tags") as batch_op:
        batch_op.alter_column(column_name="obj_name", type_=sa.Integer())
