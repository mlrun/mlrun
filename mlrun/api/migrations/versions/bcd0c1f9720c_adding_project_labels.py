"""Adding project labels

Revision ID: bcd0c1f9720c
Revises: f4249b4ba6fa
Create Date: 2020-12-20 03:42:02.763802

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "bcd0c1f9720c"
down_revision = "f4249b4ba6fa"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "projects_labels",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("value", sa.String(), nullable=True),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent"],
            ["projects.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", "parent", name="_projects_labels_uc"),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("projects_labels")
    # ### end Alembic commands ###
