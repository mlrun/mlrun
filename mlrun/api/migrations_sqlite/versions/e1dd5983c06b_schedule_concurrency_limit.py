"""Schedule concurrency limit

Revision ID: e1dd5983c06b
Revises: bcd0c1f9720c
Create Date: 2021-03-15 13:36:18.703619

"""
import sqlalchemy as sa
from alembic import op

from mlrun.config import config

# revision identifiers, used by Alembic.
revision = "e1dd5983c06b"
down_revision = "bcd0c1f9720c"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("schedules_v2") as batch_op:
        batch_op.add_column(
            sa.Column(
                "concurrency_limit",
                sa.Integer(),
                nullable=False,
                server_default=str(config.httpdb.scheduling.default_concurrency_limit),
            )
        )


def downgrade():
    with op.batch_alter_table("schedules_v2") as batch_op:
        batch_op.drop_column("concurrency_limit")
