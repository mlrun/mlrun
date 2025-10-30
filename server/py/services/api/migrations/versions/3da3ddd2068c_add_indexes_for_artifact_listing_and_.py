# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add indexes for artifact listing and tag lookup

Revision ID: 3da3ddd2068c
Revises: 31d54cd9ff11
Create Date: 2025-10-30 16:45:07.195227
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "3da3ddd2068c"
down_revision = "31d54cd9ff11"
branch_labels = None
depends_on = None


def upgrade():
    # Drop old index if it already exists (MySQL 8+ / Postgres safe)
    op.execute("DROP INDEX idx_project_bi_updated ON artifacts_v2;")

    # Recreate with new definition (adds kind + id)
    op.create_index(
        "idx_project_bi_updated",
        "artifacts_v2",
        ["project", "best_iteration", "kind", "updated", "id"],
        unique=False,
    )

    # Add new index for tag lookup
    op.create_index(
        "idx_artifacts_tags_name_obj",
        "artifacts_v2_tags",
        ["name", "obj_id"],
        unique=False,
    )


def downgrade():
    # Drop new indexes
    op.drop_index("idx_project_bi_updated", table_name="artifacts_v2")
    op.drop_index("idx_artifacts_tags_name_obj", table_name="artifacts_v2_tags")

    op.create_index(
        "idx_project_bi_updated",
        "artifacts_v2",
        ["project", "best_iteration", "updated"],
        unique=False,
    )
