# Copyright 2023 Iguazio
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

from typing import Optional, Union

from .mysql import ApplicationResultTable as MySQLApplicationResultTable
from .mysql import ModelEndpointsTable as MySQLModelEndpointsTable
from .mysql import MonitoringSchedulesTable as MySQLMonitoringSchedulesTable
from .sqlite import ApplicationResultTable as SQLiteApplicationResultTable
from .sqlite import ModelEndpointsTable as SQLiteModelEndpointsTable
from .sqlite import MonitoringSchedulesTable as SQLiteMonitoringSchedulesTable


def _get_model_endpoints_table(
    connection_string: Optional[str] = None,
) -> Union[type[MySQLModelEndpointsTable], type[SQLiteModelEndpointsTable]]:
    """Return ModelEndpointsTable based on the provided connection string"""
    if connection_string and "mysql:" in connection_string:
        return MySQLModelEndpointsTable
    return SQLiteModelEndpointsTable


def _get_application_result_table(
    connection_string: Optional[str] = None,
) -> Union[type[MySQLApplicationResultTable], type[SQLiteApplicationResultTable]]:
    """Return ModelEndpointsTable based on the provided connection string"""
    if connection_string and "mysql:" in connection_string:
        return MySQLApplicationResultTable
    return SQLiteApplicationResultTable


def _get_monitoring_schedules_table(
    connection_string: Optional[str] = None,
) -> Union[type[MySQLMonitoringSchedulesTable], type[SQLiteMonitoringSchedulesTable]]:
    """Return ModelEndpointsTable based on the provided connection string"""
    if connection_string and "mysql:" in connection_string:
        return MySQLMonitoringSchedulesTable
    return SQLiteMonitoringSchedulesTable
