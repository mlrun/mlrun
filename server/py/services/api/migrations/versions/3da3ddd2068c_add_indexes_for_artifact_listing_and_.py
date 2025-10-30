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
    op.execute(
        "CREATE INDEX idx_project_bi_updated ON artifacts_v2 (project, best_iteration, kind, updated DESC, id DESC);"
    )
    op.execute(
        "CREATE INDEX idx_artifacts_tags_name_obj ON artifacts_v2_tags (name, obj_id);"
    )


def downgrade():
    op.execute("DROP INDEX idx_project_bi_updated ON artifacts_v2;")
    op.execute("DROP INDEX idx_artifacts_tags_name_obj ON artifacts_v2_tags;")
