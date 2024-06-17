# Copyright 2024 Iguazio
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
"""Index ArtifactsV2 fields

Revision ID: 311d8cbc42a9
Revises: c0e342d73bd0
Create Date: 2024-06-16 16:00:33.418577

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "311d8cbc42a9"
down_revision = "c0e342d73bd0"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index(
        "idx_artifacts_producer_id_best_iteration_and_project",
        "artifacts_v2",
        ["project", "producer_id", "best_iteration"],
        unique=False,
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(
        "idx_artifacts_producer_id_best_iteration_and_project",
        table_name="artifacts_v2",
    )
    # ### end Alembic commands ###
