"""Index artifacts_v2_tags fields

Revision ID: 23f77f5bb5ef
Revises: ebf6f5af763d
Create Date: 2024-06-18 19:19:31.502723

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "23f77f5bb5ef"
down_revision = "ebf6f5af763d"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index(
        "idx_artifacts_v2_tags_project_name_obj_name",
        "artifacts_v2_tags",
        ["project", "name", "obj_name"],
        unique=False,
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(
        "idx_artifacts_v2_tags_project_name_obj_name",
        table_name="artifacts_v2_tags",
    )
    # ### end Alembic commands ###
