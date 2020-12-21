import os
import mlrun


class OfflineVectorResponse:
    def __init__(self, merger):
        self._merger = merger
        self.vector = merger.vector

    @property
    def status(self):
        return self._merger.get_status()

    def to_dataframe(self):
        if self.status != "ready":
            raise FeatureVectorError("feature vector dataset is not ready")
        return self._merger.get_df()

    def to_parquet(self, target_path, **kw):
        return self._upload(target_path, "parquet", **kw)

    def to_csv(self, target_path, **kw):
        return self._upload(target_path, "csv", **kw)

    def _upload(self, target_path, format="parquet", **kw):
        df = self._merger.get_df()
        data_stores = mlrun.get_data_stores()

        if format in ["csv", "parquet"]:
            writer_string = "to_{}".format(format)
            saving_func = getattr(df, writer_string, None)
            target = target_path
            to_upload = False
            if "://" in target:
                target = mktemp()
                to_upload = True
            else:
                dir = os.path.dirname(target)
                if dir:
                    os.makedirs(dir, exist_ok=True)

            saving_func(target, **kw)
            if to_upload:
                data_stores.object(url=target_path).upload(target)
                os.remove(target)
            return target_path

        raise ValueError(f"format {format} not implemented yes")
