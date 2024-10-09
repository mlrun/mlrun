import multiprocessing
from _queue import Empty

import taosws

import mlrun.errors
from mlrun.utils import logger


class QueryResult:
    def __init__(self, data, fields):
        self.data = data
        self.fields = fields

    def __eq__(self, other):
        return self.data == other.data and self.fields == other.fields


class Field:
    def __init__(self, name, type, bytes):
        self.name = name
        self.type = type
        self.bytes = bytes

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.type == other.type
            and self.bytes == other.bytes
        )


class Statement:
    def __init__(self, function, kwargs):
        self.function = function
        self.kwargs = kwargs

    def prepare(self, statement):
        return self.function(statement, **self.kwargs)


class TDEngineConnection:
    def __init__(self, connection_string):
        self._connection_string = connection_string
        self.prefix_statements = []

    def _run(self, q, statements, query):
        try:
            conn = taosws.connect(self._connection_string)

            for statement in self.prefix_statements + statements:
                if isinstance(statement, Statement):
                    prepared_statement = statement.prepare(conn.statement())
                    prepared_statement.execute()
                else:
                    conn.execute(statement)

            if not query:
                q.put(None)
                return

            res = conn.query(query)

            # taosws.TaosField is not serializable
            fields = [
                Field(field.name(), field.type(), field.bytes()) for field in res.fields
            ]

            q.put(QueryResult(list(res), fields))
        except Exception as e:
            q.put(e)

    def run(self, statements=None, query=None, retries=2, timeout=5):
        mp = multiprocessing.get_context("spawn")
        statements = statements or []
        if not isinstance(statements, list):
            statements = [statements]
        overall_retries = retries
        while retries >= 0:
            q = mp.Queue()
            process = mp.Process(target=self._run, args=[q, statements, query])
            try:
                process.start()
                process.join(timeout=timeout)
                try:
                    result = q.get(timeout=0)
                    if isinstance(result, Exception):
                        raise result
                    return result
                except Empty:
                    logger.warn(
                        f"TDEngine statements {statements} and query '{query}' timed out after {timeout} seconds."
                        f"{retries} retries left."
                    )
                    if retries == 0:
                        raise mlrun.errors.MLRunTimeoutError(
                            f"TDEngine statements {statements} and query '{query}' timed out after {timeout} seconds "
                            f"and {overall_retries} retries"
                        )
                    retries -= 1
            finally:
                process.close()
