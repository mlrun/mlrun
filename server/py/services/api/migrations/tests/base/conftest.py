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
import functools
import logging
from datetime import datetime

import pytest
import sqlalchemy.orm


def pytest_configure(config):
    logging.getLogger("faker.factory").setLevel(logging.WARNING)


@pytest.fixture
def alembic_session(alembic_engine):
    session_class = sqlalchemy.orm.sessionmaker(bind=alembic_engine)
    session = session_class()
    try:
        yield session
    finally:
        session.close()


class FrozenDatetime(datetime):
    """`datetime` subclass whose `now()` returns a configurable constant."""

    _frozen_now = datetime(1970, 1, 1)

    @classmethod  # type: ignore[override]
    def now(cls, tz=None):
        return cls._frozen_now.replace(tzinfo=tz)


def freeze_datetime(target_dt: datetime):
    """Decorator that temporarily freezes `datetime.now()` to *target_dt*."""

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            monkey = pytest.MonkeyPatch()
            try:
                FrozenDatetime._frozen_now = target_dt
                monkey.setattr(
                    "services.api.utils.db.partitioner.datetime", FrozenDatetime
                )
                return test_func(*args, **kwargs)
            finally:
                monkey.undo()

        return wrapper

    return decorator
