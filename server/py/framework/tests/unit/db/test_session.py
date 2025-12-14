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
import unittest.mock

import pytest
import sqlalchemy.orm

import framework.db.session
from framework.tests.unit.db.common_fixtures import TestDatabaseBase


class TestRunAsyncFunctionWithNewDbSession(TestDatabaseBase):
    @pytest.mark.asyncio
    async def test_run_async_function_with_new_db_session_async_func(self):
        """Test that async functions get a new session and execute correctly"""
        session_received = None
        result_value = "test_result"

        async def async_func(session, *args, **kwargs):
            nonlocal session_received
            session_received = session
            assert isinstance(session, sqlalchemy.orm.Session)
            return result_value

        result = await framework.db.session.run_async_function_with_new_db_session(
            async_func, "arg1", kwarg1="value1"
        )

        assert result == result_value
        assert session_received is not None
        assert isinstance(session_received, sqlalchemy.orm.Session)

    @pytest.mark.asyncio
    async def test_run_async_function_with_new_db_session_sync_func(self):
        """Test that sync functions are run in threadpool with a new session"""
        session_received = None
        result_value = "test_result"

        def sync_func(session, *args, **kwargs):
            nonlocal session_received
            session_received = session
            assert isinstance(session, sqlalchemy.orm.Session)
            return result_value

        result = await framework.db.session.run_async_function_with_new_db_session(
            sync_func, "arg1", kwarg1="value1"
        )

        assert result == result_value
        assert session_received is not None
        assert isinstance(session_received, sqlalchemy.orm.Session)

    @pytest.mark.asyncio
    async def test_run_async_function_with_new_db_session_async_func_with_exception(
        self,
    ):
        """Test that exceptions in async functions are properly propagated"""
        error_msg = "Test error"

        async def async_func(session, *args, **kwargs):
            raise ValueError(error_msg)

        with pytest.raises(ValueError, match=error_msg):
            await framework.db.session.run_async_function_with_new_db_session(
                async_func
            )

    @pytest.mark.asyncio
    async def test_run_async_function_with_new_db_session_sync_func_with_exception(
        self,
    ):
        """Test that exceptions in sync functions are properly propagated"""
        error_msg = "Test error"

        def sync_func(session, *args, **kwargs):
            raise ValueError(error_msg)

        with pytest.raises(ValueError, match=error_msg):
            await framework.db.session.run_async_function_with_new_db_session(sync_func)

    @pytest.mark.asyncio
    async def test_run_async_function_with_new_db_session_session_isolation(self):
        """Test that each call gets a different session"""
        sessions_received = []

        async def async_func(session, *args, **kwargs):
            sessions_received.append(session)
            return "result"

        # Call twice concurrently
        results = await asyncio.gather(
            framework.db.session.run_async_function_with_new_db_session(async_func),
            framework.db.session.run_async_function_with_new_db_session(async_func),
        )

        assert len(sessions_received) == 2
        # Sessions should be different objects
        assert sessions_received[0] is not sessions_received[1]
        assert results == ["result", "result"]

    @pytest.mark.asyncio
    async def test_run_async_function_with_new_db_session_args_kwargs_passed(self):
        """Test that args and kwargs are properly passed to the function"""
        received_args = None
        received_kwargs = None

        async def async_func(session, *args, **kwargs):
            nonlocal received_args, received_kwargs
            received_args = args
            received_kwargs = kwargs
            return "result"

        await framework.db.session.run_async_function_with_new_db_session(
            async_func, "arg1", "arg2", kwarg1="value1", kwarg2="value2"
        )

        assert received_args == ("arg1", "arg2")
        assert received_kwargs == {"kwarg1": "value1", "kwarg2": "value2"}

    @pytest.mark.asyncio
    async def test_run_async_function_with_new_db_session_session_cleanup_on_error(
        self,
    ):
        """Test that session context manager properly handles cleanup on error"""
        session = None

        async def async_func(session_, *args, **kwargs):
            # Verify session is valid before error
            assert isinstance(session_, sqlalchemy.orm.Session)
            session_.rollback = unittest.mock.MagicMock()

            nonlocal session
            session = session_
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await framework.db.session.run_async_function_with_new_db_session(
                async_func
            )
        assert session.rollback.called, "Session rollback should be called on error"
