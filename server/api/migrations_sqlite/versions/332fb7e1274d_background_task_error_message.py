"""Background task error message

Revision ID: 332fb7e1274d
Revises: bf91ff18513b
Create Date: 2023-11-08 10:56:54.339846

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "332fb7e1274d"
down_revision = "bf91ff18513b"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "background_tasks", sa.Column("error", sa.String(length=255), nullable=True)
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("background_tasks", "error")
    # ### end Alembic commands ###
