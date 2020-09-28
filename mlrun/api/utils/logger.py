import logging


class StatusFilterStreamHandler(logging.StreamHandler):
    """
    Custom Stream Handler specialized for uvicorn's access logger to filter out log messages based on the status code
    This handler will print log messages only for logs related to responses with status code which outside of [200,299]
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "status_code") and isinstance(record.status_code, int):
            if 200 <= record.status_code <= 299:
                return False
        return super().filter(record)
