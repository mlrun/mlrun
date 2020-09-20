import importlib.resources
import json

import mlrun.utils.singleton


class Version(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self.version_info = json.loads(
            importlib.resources.read_text("mlrun.utils.version", "version.json")
        )

    def get(self):
        return self.version_info
