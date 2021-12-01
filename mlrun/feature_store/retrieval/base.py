import abc

from mlrun.datastore.targets import CSVTarget, ParquetTarget


class BaseMerger(abc.ABC):
    """abstract feature merger class"""

    @abc.abstractmethod
    def __init__(self, vector, **engine_args):
        self._result_df = None
        self.vector = vector

    @abc.abstractmethod
    def start(
        self,
        entity_rows=None,
        entity_timestamp_column=None,
        target=None,
        drop_columns=None,
        start_time=None,
        end_time=None,
        with_indexes=None,
        update_stats=None,
    ):
        raise NotImplementedError("start() operation not supported in class")

    def get_status(self):
        if self._result_df is None:
            raise RuntimeError("unexpected status, no result df")
        return "completed"

    def get_df(self):
        return self._result_df

    def to_parquet(self, target_path, **kw):
        """return results as parquet file"""
        size = ParquetTarget(path=target_path).write_dataframe(self.get_df(), **kw)
        return size

    def to_csv(self, target_path, **kw):
        """return results as csv file"""
        size = CSVTarget(path=target_path).write_dataframe(self.get_df(), **kw)
        return size
