"""add publish_time to feature_set

Revision ID: 233a56776ddb
Revises: 9d16de5f03a7
Create Date: 2021-11-08 13:55:18.850878

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '233a56776ddb'
down_revision = '9d16de5f03a7'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('feature_sets', sa.Column('publish_time', sa.TIMESTAMP(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('feature_sets', 'publish_time')
    # ### end Alembic commands ###
