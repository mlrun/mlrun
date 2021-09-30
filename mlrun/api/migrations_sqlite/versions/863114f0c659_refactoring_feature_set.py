"""Refactoring feature set

Revision ID: 863114f0c659
Revises: f7b5a1a03629
Create Date: 2020-11-11 11:22:36.653049

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "863114f0c659"
down_revision = "1c954f8cb32d"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("feature_sets") as batch_op:
        batch_op.add_column(sa.Column("object", sa.JSON(), nullable=True))
        batch_op.drop_column("status")


def downgrade():
    with op.batch_alter_table("feature_sets") as batch_op:
        batch_op.add_column(sa.Column("status", sa.JSON(), nullable=True))
        batch_op.drop_column("object")
