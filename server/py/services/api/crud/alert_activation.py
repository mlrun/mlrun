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
#


from sqlalchemy.orm import Session

import mlrun.common.schemas.alert
import mlrun.common.schemas.partition
import mlrun.utils.singleton
import services.api.utils.db.partitioner


class AlertActivation(
    metaclass=mlrun.utils.singleton.Singleton,
):
    def create_and_drop_partitions(
        self,
        session: Session,
        retention_days: int,
    ) -> None:
        """
        Creates partitions for the future based on the specified retention days
        and drops old partitions that are older than the retention period.

        :param session: SQLAlchemy session for database operations.
        :param retention_days: The number of days to retain partitions.
        """

        services.api.utils.db.partitioner.MySQLPartitioner().create_and_drop_partitions(
            session=session,
            retention_days=retention_days,
            table_name=self.table_name,
        )

    @property
    def table_name(self) -> str:
        return "alert_activation"
