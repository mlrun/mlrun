"""schedule id

Revision ID: cf21882f938e
Revises: 11f8dd2dc9fe
Create Date: 2020-10-07 11:21:49.223077

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "cf21882f938e"
down_revision = "11f8dd2dc9fe"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("schedules_v2") as batch_op:
        batch_op.add_column(sa.Column("id", sa.Integer(), nullable=False))
        batch_op.create_primary_key("pk_schedules_v2", ["id"])
        batch_op.create_unique_constraint("_schedules_v2_uc", ["project", "name"])


def downgrade():
    with op.batch_alter_table("schedules_v2") as batch_op:
        batch_op.drop_constraint("_schedules_v2_uc", type_="unique")
        batch_op.create_primary_key("pk_schedules_v2", ["name", "project"])
        batch_op.drop_column("id")
