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

import os
from pathlib import Path

import yaml

import mlrun.utils
from mlrun.agentic.chains.base import ChainRunner
from mlrun.agentic.chains.declarative.agent import DeclarativeAgent
from mlrun.agentic.chains.declarative.team import DeclarativeTeam
from mlrun.agentic.schemas import WorkflowEvent

logger = mlrun.utils.logger


class DeclarativeRunner(ChainRunner):
    """Factory ChainRunner that loads ARK YAMLs and delegates to the right handler.

    Supports all input forms:
    - Single file path (including multi-document YAML with ``---`` separators)
    - List of file paths or raw YAML strings
    - Directory path (loads all ``*.yaml`` / ``*.yml`` files)
    - Raw YAML string (single or multi-document)

    Classifies loaded documents by ``kind`` field:
    - ``kind: Team`` becomes the primary orchestrator (delegates to DeclarativeTeam).
    - ``kind: Agent`` documents become members (or the primary if no Team is found).

    :param yaml_input: File path, directory path, raw YAML string, or list thereof.
    """

    def __init__(self, yaml_input, **kwargs):
        super().__init__(**kwargs)
        self.yaml_input = yaml_input
        self._delegate = None

    def post_init(self, mode="sync", context=None, namespace=None, **kwargs):
        """Load YAMLs, classify by kind, build the appropriate delegate."""
        documents = self._load_yamls(self.yaml_input)
        if not documents:
            raise ValueError("No YAML documents loaded from input")

        primary, members_by_name = self._classify_yamls(documents)

        kind = primary.get("kind", "Agent")
        if kind == "Team":
            self._delegate = DeclarativeTeam(
                team_config=primary,
                members_by_name=members_by_name,
            )
        else:
            self._delegate = DeclarativeAgent(agent_config=primary)

        self._delegate.post_init(
            mode=mode, context=context, namespace=namespace, **kwargs
        )
        logger.info("DeclarativeRunner initialized", kind=kind)

    def _run(self, event: WorkflowEvent):
        """Delegate execution to the resolved handler."""
        return self._delegate._run(event)

    @staticmethod
    def _load_yamls(yaml_input) -> list[dict]:
        """Load YAML documents from various input forms.

        :param yaml_input: File path, directory, raw YAML string, or list thereof.
        :return: List of parsed YAML dicts.
        """
        if isinstance(yaml_input, list):
            documents = []
            for item in yaml_input:
                documents.extend(DeclarativeRunner._load_single(item))
            return documents

        return DeclarativeRunner._load_single(yaml_input)

    @staticmethod
    def _load_single(item: str) -> list[dict]:
        """Load YAML documents from a single input (file, directory, or raw string).

        :param item: A file path, directory path, or raw YAML string.
        :return: List of parsed YAML dicts.
        """
        path = Path(item) if not item.lstrip().startswith("{") else None

        # Check if it's a file path
        if path and path.is_file():
            return DeclarativeRunner._load_file(path)

        # Check if it's a directory
        if path and path.is_dir():
            documents = []
            for yaml_file in sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml")):
                documents.extend(DeclarativeRunner._load_file(yaml_file))
            return documents

        # Treat as raw YAML string (handles both single and multi-document)
        return DeclarativeRunner._parse_yaml_string(item)

    @staticmethod
    def _load_file(file_path: Path) -> list[dict]:
        """Load all YAML documents from a file (supports multi-document ``---``).

        :param file_path: Path to a YAML file.
        :return: List of parsed YAML dicts.
        """
        with open(file_path) as f:
            content = f.read()
        return DeclarativeRunner._parse_yaml_string(content)

    @staticmethod
    def _parse_yaml_string(content: str) -> list[dict]:
        """Parse a YAML string that may contain multiple documents separated by ``---``.

        :param content: Raw YAML string.
        :return: List of parsed YAML dicts (skips None documents).
        """
        documents = []
        for doc in yaml.safe_load_all(content):
            if doc is not None:
                documents.append(doc)
        return documents

    @staticmethod
    def _classify_yamls(documents: list[dict]) -> tuple[dict, dict[str, dict]]:
        """Classify YAML documents into a primary document and member agents.

        Rules:
        - ``kind: Team`` becomes the primary; all ``kind: Agent`` docs become members.
        - If no Team is found, the first Agent is the primary.

        :param documents: List of parsed YAML dicts.
        :return: Tuple of (primary_config, members_by_name).
        """
        team = None
        agents = {}

        for doc in documents:
            kind = doc.get("kind", "Agent")
            name = doc.get("metadata", {}).get("name", "")

            if kind == "Team":
                if team is not None:
                    logger.warning("Multiple Team documents found, using the first one")
                    continue
                team = doc
            else:
                if not name:
                    name = f"agent_{len(agents)}"
                agents[name] = doc

        if team:
            return team, agents

        # No team — first agent is primary
        if not agents:
            raise ValueError("No Agent or Team documents found in YAML input")

        first_name = next(iter(agents))
        primary = agents.pop(first_name)
        return primary, agents

    @staticmethod
    def _is_likely_path(item: str) -> bool:
        """Heuristic to check if a string looks like a file/directory path.

        :param item: String to check.
        :return: True if it looks like a path.
        """
        return os.sep in item or item.endswith((".yaml", ".yml"))
