"""add publish_time to feature set

Revision ID: 63a856bce322
Revises: accf9fc83d38
Create Date: 2021-11-08 13:36:42.166875

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '63a856bce322'
down_revision = 'accf9fc83d38'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("feature_sets") as batch_op:
        batch_op.add_column(sa.Column('publish_time', sa.TIMESTAMP(), nullable=True))


def downgrade():
    with op.batch_alter_table("feature_sets") as batch_op:
        batch_op.drop_column('publish_time')
