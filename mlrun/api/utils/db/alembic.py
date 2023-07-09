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
import pathlib
import typing

import alembic.command
import alembic.config

from mlrun.utils import logger


class AlembicUtil(object):
    def __init__(
        self, alembic_config_path: pathlib.Path, data_version_is_latest: bool = True
    ):
        self._alembic_config_path = str(alembic_config_path)
        self._alembic_config = alembic.config.Config(self._alembic_config_path)
        self._alembic_output = ""
        self._data_version_is_latest = data_version_is_latest
        self._revision_history = self._get_revision_history_list()
        self._latest_revision = self._revision_history[0]
        self._initial_revision = self._revision_history[-1]

    def init_alembic(self):
        logger.debug("Performing alembic schema migrations")
        alembic.command.upgrade(self._alembic_config, "head")

    def is_schema_migration_needed(self):
        current_revision = self._get_current_revision()
        return current_revision != self._latest_revision

    def is_migration_from_scratch(self):
        current_revision = self._get_current_revision()
        if not current_revision:
            return True
        return current_revision == self._initial_revision

    def _get_current_revision(self) -> typing.Optional[str]:

        # create separate config in order to catch the stdout
        catch_stdout_config = alembic.config.Config(self._alembic_config_path)
        catch_stdout_config.print_stdout = self._save_output

        self._flush_output()
        try:
            alembic.command.current(catch_stdout_config)
            return self._alembic_output.strip().replace(" (head)", "")
        except Exception as exc:
            if "Can't locate revision identified by" in exc.args[0]:

                # DB has a revision that isn't known to us, extracting it from the exception.
                return exc.args[0].split("'")[2]

            return None

    def _get_revision_history_list(self) -> typing.List[str]:
        """
        Returns a list of the revision history sorted from latest to oldest.
        """

        # create separate config in order to catch the stdout
        catch_stdout_config = alembic.config.Config(self._alembic_config_path)
        catch_stdout_config.print_stdout = self._save_output

        self._flush_output()
        alembic.command.history(catch_stdout_config)
        return self._parse_revision_history(self._alembic_output)

    @staticmethod
    def _parse_revision_history(output: str) -> typing.List[str]:
        return [line.split(" ")[2].replace(",", "") for line in output.splitlines()]

    def _save_output(self, text: str, *_):
        self._alembic_output += f"{text}\n"

    def _flush_output(self):
        self._alembic_output = ""
