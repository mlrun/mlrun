import os
import pathlib
from tempfile import mktemp
from .model import TargetTypes


def write_to_target_store(kind, source, target_path, data_stores):
    """write/ingest data to a target store"""
    if kind == TargetTypes.parquet:
        return upload(source, target_path, data_stores)


def upload(source, target_path, data_stores, format="parquet", **kw):
    suffix = pathlib.Path(target_path).suffix
    if not suffix:
        target_path = target_path + "." + format
    if isinstance(source, str):
        if source and os.path.isfile(source):
            data_stores.object(url=target_path).upload(source)
        return target_path

    df = source
    if df is None:
        return None

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
