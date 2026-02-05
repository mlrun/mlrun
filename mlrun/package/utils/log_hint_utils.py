# Copyright 2023 Iguazio
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

from mlrun.errors import MLRunInvalidArgumentError


class LogHintKey:
    """
    Known keys for a log hint to have.
    """

    KEY = "key"
    ARTIFACT_TYPE = "artifact_type"
    EXTRA_DATA = "extra_data"
    METRICS = "metrics"


class LogHintUtils:
    """
    Static class for utilities functions to process log hints.
    """

    @staticmethod
    def parse_log_hint(
        log_hint: dict[str, str] | str | None,
    ) -> dict[str, str] | None:
        """
        Parse a given log hint from string to a logging configuration dictionary. The string will be read as the
        artifact key ('key' in the dictionary) and if the string have a single colon, the following structure is
        assumed: "<artifact_key> : <artifact_type>".

        If a logging configuration dictionary is received, it will be validated to have a key field.

        None will be returned as None.

        :param log_hint: The log hint to parse.

        :return: The hinted logging configuration.

        :raise MLRunInvalidArgumentError: In case the log hint is not following the string structure or the dictionary
                                          is missing the key field.
        """
        # Check for None value:
        if log_hint is None:
            return None

        # If the log hint was provided as a string, construct a dictionary out of it:
        if isinstance(log_hint, str):
            # Check if only key is given:
            if ":" not in log_hint:
                log_hint = {LogHintKey.KEY: log_hint}
            # Check for valid "<key> : <artifact type>" pattern:
            else:
                if log_hint.count(":") > 1:
                    raise MLRunInvalidArgumentError(
                        f"Incorrect log hint pattern. Log hints can have only a single ':' in them to specify the "
                        f"desired artifact type the returned value will be logged as: "
                        f"'<artifact_key> : <artifact_type>', but given: {log_hint}"
                    )
                # Split into key and type:
                key, artifact_type = log_hint.replace(" ", "").split(":")
                if artifact_type == "":
                    raise MLRunInvalidArgumentError(
                        f"Incorrect log hint pattern. The ':' in a log hint should specify the desired artifact type "
                        f"the returned value will be logged as in the following pattern: "
                        f"'<artifact_key> : <artifact_type>', but no artifact type was given: {log_hint}"
                    )
                log_hint = {
                    LogHintKey.KEY: key,
                    LogHintKey.ARTIFACT_TYPE: artifact_type,
                }

        # Validate the log hint dictionary has the mandatory key:
        if LogHintKey.KEY not in log_hint:
            raise MLRunInvalidArgumentError(
                f"A log hint dictionary must include the 'key' - the artifact key (it's name). The following log hint "
                f"is missing the key: {log_hint}."
            )

        return log_hint

    @staticmethod
    def extract_unbundling_from_key(log_hint: str) -> tuple[str, bool | int]:
        """
        Extract unbundling information from a log hint key if exists. If the log hint key contains an asterisk '*', it
        indicates that unbundling is required. The part before the asterisk represents the unbundle level (an integer or
        empty for full unbundling), and the part after the asterisk is the actual artifact key.

        :param log_hint: The log hint key to extract unbundling information from.

        :return: A tuple containing the actual artifact key and the unbundle level (True for full unbundling, False for
                 no unbundling, or an integer for specific unbundle level).
        """
        # Check if unbundling is required:
        if "*" not in log_hint:
            return log_hint, False

        # Extract unbundle level and key:
        unbundle_level, key = log_hint.split("*", 1)

        # Make sure a key is given:
        if not key.strip():
            raise MLRunInvalidArgumentError(
                f"Invalid log hint key '{log_hint}'. Key is missing after the '*' indicating unbundling. A log hint "
                f"key with unbundling should be in the format of "
                f"'<unbundle_level>*<key>' or '*<key>' for full "
                f"unbundling."
            )

        # If unbundle level is given, convert to int:
        if unbundle_level.strip():
            try:
                unbundle_level = int(unbundle_level.strip())
            except ValueError:
                raise MLRunInvalidArgumentError(
                    f"Invalid unbundle level '{unbundle_level}' in log hint '{log_hint}'. "
                    f"Unbundle level must be an integer."
                )
        else:
            # If no level is given, set to True for full unbundling:
            unbundle_level = True

        return key.strip(), unbundle_level
