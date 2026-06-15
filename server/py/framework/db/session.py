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

import asyncio
import contextlib
import inspect

from sqlalchemy.orm import Session

import mlrun.utils

import framework.db.sqldb.sql_session


def create_session() -> Session:
    return framework.db.sqldb.sql_session.create_session()


def close_session(db_session):
    db_session.close()


def run_function_with_new_db_session(func, *args, **kwargs):
    """
    Run a function with a new db session, useful for concurrent requests where we can't share a single session.
    However, any changes made by the new session will not be visible to old sessions until the old sessions commit
    due to isolation level.
    """
    with get_db_session(commit=False) as session:
        return func(session, *args, **kwargs)


async def run_async_function_with_new_db_session(func, *args, **kwargs):
    """
    Run an async function with a new db session.
    If the func is a coroutine function (async def), use run_async_function_with_new_db_session below.
    alternatively, given the async context, run the synchronous function in a thread pool.

    Any changes made by the new session will not be visible to old sessions until the old sessions commit
    due to isolation level.
    """

    # function is async. wrap its execution with a new db session
    if inspect.iscoroutinefunction(func):
        # commit is set to False as this function lets the caller handle committing/rolling back the session
        async with get_db_session_async(commit=False) as session:
            return await func(session, *args, **kwargs)

    # function is sync running in async context,
    # move all together to a thread and execute it non-blocking
    return await mlrun.utils.run_in_threadpool(
        run_function_with_new_db_session,
        func,
        *args,
        **kwargs,
    )


@contextlib.asynccontextmanager
async def get_db_session_async(commit=True):
    """
    Async context manager that provides a database session and handles commit/rollback.
    :param commit: Whether to commit the session on successful completion. Defaults to True.
    """
    session = await asyncio.to_thread(create_session)
    try:
        yield session
        if commit:
            await asyncio.to_thread(session.commit)
    except Exception:
        await asyncio.to_thread(session.rollback)
        raise
    finally:
        await asyncio.to_thread(session.close)


@contextlib.contextmanager
def get_db_session(commit=True):
    """
    Context manager that provides a database session and handles commit/rollback.

    :param commit: Whether to commit the session on successful completion. Defaults to True.
    """
    session = create_session()
    try:
        yield session
        if commit:
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
