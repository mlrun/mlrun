"""Background task error message

Revision ID: 0a6ac170f91d
Revises: 27ed4ecb734c
Create Date: 2023-11-06 09:06:48.928402

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0a6ac170f91d"
down_revision = "27ed4ecb734c"
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
