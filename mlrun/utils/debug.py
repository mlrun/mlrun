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

import inspect
from collections.abc import Callable
from typing import Any

import mlrun
from mlrun.utils import logger

# "artifactory.iguazeng.com:10557/mlrun:unstable"


def debug_info(data, logger=None, msg=""):
    """
    Log debug information including stack trace, dictionary contents, and environment variables
    using JSON formatting. Handles non-JSON-serializable objects by replacing them with 'Unknown'.

    Args:
        data: Dictionary containing debug data
        logger: Optional logger instance. If None, print will be used
        msg: Optional message providing context for the debug information
    from mlrun.utils.debug import  debug_info
    debug_info({})
    """
    import json
    import traceback

    def make_json_serializable(obj):
        """Convert a non-JSON-serializable object to a serializable format."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return "Unknown"

    debug_data = {
        "message": f"KUKAREKU=={msg}",
        "stack_trace": [],
        "debug_data": make_json_serializable(data),
        "environment": {},
    }

    # Get the current stack trace
    stack_trace = traceback.format_stack()
    # Remove the last entry which is this function call
    debug_data["stack_trace"] = [line.strip() for line in stack_trace[:-1]]

    # Add environment variables without masking
    # debug_data["environment"] = dict(sorted(os.environ.items()))

    # Convert to JSON string
    json_output = json.dumps(debug_data, indent=2)

    # Output using logger or print
    if logger:
        logger.error(json_output)
    else:
        mlrun.utils.logger.error(json_output)


def traced_call(func: Callable, *args, **kwargs) -> Any:
    name = f"{func.__module__}.{func.__name__}"
    formatted_args = _format_args(func, args, kwargs)

    logger.info(f"TDECALL: {name}({formatted_args})")

    try:
        result = func(*args, **kwargs)
        result_repr = "None" if result is None else _repr(result)
        logger.info(f"TDERETURN: {name} -> {result_repr}")
        return result

    except Exception as e:
        logger.info(f"TDEEXCEPTION: {name} -> {type(e).__name__}: {str(e)[:100]}")
        raise


def _format_args(func, args, kwargs):
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        parts = []
        for name, value in bound.arguments.items():
            if name not in ("self", "cls"):
                parts.append(f"{name}={_repr(value, 80)}")
        return ", ".join(parts)
    except Exception:
        parts = [
            _repr(arg, 80)
            for arg in args[1 if args and hasattr(args[0], "__dict__") else 0 :]
        ]
        parts.extend(f"{k}={_repr(v, 80)}" for k, v in kwargs.items())
        return ", ".join(parts)


def _repr(obj, max_len=200):
    try:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return repr(obj)
        elif isinstance(obj, (list, tuple)):
            if len(obj) <= 5:
                return repr(obj)
            else:
                items = [_repr(item, 50) for item in obj[:3]]
                return f"[{', '.join(items)}, +{len(obj)-3} more]"
        elif isinstance(obj, dict):
            if len(obj) <= 3:
                return repr(obj)
            else:
                items = []
                for i, (k, v) in enumerate(obj.items()):
                    if i >= 2:
                        break
                    items.append(f"{_repr(k, 30)}: {_repr(v, 50)}")
                return f"{{{', '.join(items)}, +{len(obj)-2} more}}"
        else:
            if hasattr(obj, "__dict__"):
                attrs = obj.__dict__
                if len(attrs) <= 3:
                    attr_str = ", ".join(
                        f"{k}={_repr(v, 30)}" for k, v in list(attrs.items())[:3]
                    )
                    return f"<{type(obj).__name__}({attr_str})>"
                else:
                    attr_str = ", ".join(
                        f"{k}={_repr(v, 30)}" for k, v in list(attrs.items())[:2]
                    )
                    return f"<{type(obj).__name__}({attr_str}, +{len(attrs)-2} more)>"
            else:
                return f"<{type(obj).__name__}>"
    except Exception:
        return f"<{type(obj).__name__}>"


async def traced_call_async(func: Callable, *args, **kwargs) -> Any:
    """Async version of traced_call for coroutine functions."""
    name = f"{func.__module__}.{func.__name__}"
    formatted_args = _format_args(func, args, kwargs)

    logger.info(f"TDECALL: {name}({formatted_args})")

    try:
        result = await func(*args, **kwargs)
        result_repr = "None" if result is None else _repr(result)
        logger.info(f"TDERETURN: {name} -> {result_repr}")
        return result

    except Exception as e:
        logger.info(f"TDEEXCEPTION: {name} -> {type(e).__name__}: {str(e)[:100]}")
        raise


def wrap_object_with_tracing(
    obj,
    include_private=False,
    exclude_methods=None,
    include_patterns=None,
    exclude_patterns=None,
    condition_func=None,
):
    """
    Dynamically wrap all methods of an object with tracing, supporting both sync and async methods.

    Args:
        obj: The object whose methods to wrap
        include_private: Whether to wrap private methods (starting with _)
        exclude_methods: Set of method names to exclude from wrapping
        include_patterns: List of regex patterns - only wrap methods matching these
        exclude_patterns: List of regex patterns - exclude methods matching these
        condition_func: Function that returns True/False whether to apply wrapping

    Returns:
        The same object with wrapped methods

    Example:
        # Basic usage - wrap all public methods
        connection = TimescaleDBConnectionIn(dsn)
        wrap_object_with_tracing(connection)

        # Include private methods
        wrap_object_with_tracing(connection, include_private=True)

        # Exclude specific methods
        wrap_object_with_tracing(connection, exclude_methods={'__init__', '_cleanup_connection'})

        # Use patterns to be more selective
        wrap_object_with_tracing(
            connection,
            include_patterns=[r'_execute.*', r'run'],  # Only wrap _execute* and run methods
            exclude_patterns=[r'.*cleanup.*']          # But exclude anything with 'cleanup'
        )

        # Conditional wrapping based on environment
        import os
        wrap_object_with_tracing(
            connection,
            condition_func=lambda: os.getenv('MLRUN_DEBUG_TRACE', '').lower() in ('true', '1', 'yes')
        )
    """
    import re
    import types

    # Check if wrapping should be applied
    if condition_func and not condition_func():
        return obj

    exclude_methods = exclude_methods or set()
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []

    def should_wrap_method(method_name):
        if method_name in exclude_methods:
            return False
        if not include_private and method_name.startswith("_"):
            return False
        if method_name.startswith("__") and method_name.endswith("__"):
            return False
        if include_patterns and not any(
            re.match(pattern, method_name) for pattern in include_patterns
        ):
            return False
        return not exclude_patterns or not any(
            re.match(pattern, method_name) for pattern in exclude_patterns
        )

    # Get all attributes that are methods
    for attr_name in dir(obj):
        if not should_wrap_method(attr_name):
            continue

        attr = getattr(obj, attr_name)

        # Check if it's a callable method
        if callable(attr) and hasattr(attr, "__self__"):
            original_method = attr.__func__

            # Check if the original method is async
            if inspect.iscoroutinefunction(original_method):

                def create_async_wrapper(orig_method, method_name):
                    async def async_wrapper(self, *args, **kwargs):
                        return await traced_call_async(
                            orig_method, self, *args, **kwargs
                        )

                    async_wrapper.__name__ = method_name
                    async_wrapper.__doc__ = orig_method.__doc__
                    async_wrapper.__qualname__ = (
                        f"{obj.__class__.__name__}.{method_name}"
                    )
                    return async_wrapper

                wrapped_method = create_async_wrapper(original_method, attr_name)
            else:

                def create_sync_wrapper(orig_method, method_name):
                    def sync_wrapper(self, *args, **kwargs):
                        return traced_call(orig_method, self, *args, **kwargs)

                    sync_wrapper.__name__ = method_name
                    sync_wrapper.__doc__ = orig_method.__doc__
                    sync_wrapper.__qualname__ = (
                        f"{obj.__class__.__name__}.{method_name}"
                    )
                    return sync_wrapper

                wrapped_method = create_sync_wrapper(original_method, attr_name)

            bound_wrapper = types.MethodType(wrapped_method, obj)
            setattr(obj, attr_name, bound_wrapper)

    return obj
