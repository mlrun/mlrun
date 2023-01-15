import asyncio


async def maybe_coroutine(o):
    """
    If o is a coroutine, await it and return the result. Otherwise, return results.
    This is useful for when function callee is not sure if the response should be awaited or not.
    It is required for the function callee to be async. (e.g.: async def).
    """
    if asyncio.iscoroutine(o):
        return await o
    return o
