# Copyright 2018 Iguazio
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
from mlrun.api.db.sqldb.session import create_session
from mlrun.api.utils.scheduler import Scheduler

# TODO: something nicer
scheduler: Scheduler = None


async def initialize_scheduler():
    global scheduler
    scheduler = Scheduler()
    db_session = None
    try:
        db_session = create_session()
        await scheduler.start(
            db_session,
        )
    finally:
        db_session.close()


def get_scheduler() -> Scheduler:
    global scheduler
    return scheduler
