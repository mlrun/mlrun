import importlib.resources
import json

import mlrun.utils
import mlrun.utils.singleton


class Version(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self.version_info = {
            "git_commit": "unknown",
            "version": "unstable"
        }
        try:
            self.version_info = json.loads(
                importlib.resources.read_text("mlrun.utils.version", "version.json")
            )
        except:
            mlrun.utils.logger.warning("Failed resolving version info. Ignoring and using defaults")

    def get(self):
        return self.version_info
