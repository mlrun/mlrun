# Copyright 2018 Iguazio
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
#
from pathlib import Path

from mlrun.config import config

# TODO: something nicer
logs_dir: Path = None


def get_logs_dir() -> Path:
    global logs_dir
    return logs_dir


def initialize_logs_dir():
    global logs_dir
    logs_dir = Path(config.httpdb.logs_path)
