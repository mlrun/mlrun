"""add_publish_time_to_feature_set

Revision ID: e150ebfd83dd
Revises: e5594ed3ab53
Create Date: 2022-06-01 12:28:15.024080

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e150ebfd83dd'
down_revision = 'e5594ed3ab53'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("feature_sets") as batch_op:
        batch_op.add_column(sa.Column("publish_time", sa.TIMESTAMP(), nullable=True))


def downgrade():
    with op.batch_alter_table("feature_sets") as batch_op:
        batch_op.drop_column("publish_time")
