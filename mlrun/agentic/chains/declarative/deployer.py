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

"""Deployer for ARK declarative agents and teams to MLRun serving."""

import mlrun
import mlrun.errors
import mlrun.projects
import mlrun.serving
from mlrun.agentic.chains.base import HistorySaver, SessionLoader
from mlrun.agentic.chains.declarative.agent import DeclarativeAgent
from mlrun.agentic.chains.declarative.router import DeclarativeTeamRouter
from mlrun.agentic.chains.declarative.runner import DeclarativeRunner


class AgentsAtScaleDeployer:
    """Generic deployer for ARK YAML agents and teams to MLRun serving.

    Automatically detects the YAML kind and strategy to build the appropriate topology:
    - Single Agent: Flat DeclarativeRunner step
    - Sequential Team: Linear chain of DeclarativeAgent steps (no router, no LangGraph)
    - Complex Teams (round-robin/selector/graph): DeclarativeTeamRouter with LangGraph

    Example::

        deployer = AgentsAtScaleDeployer(
            yaml_input="team.yaml",
            source_file="tools.py",
            project=project,
        )
        serving_fn = deployer.build(name="my-team", with_session=True)
        deployer.create_mock()
        result = deployer.infer("What's the weather in NYC?")

    :param yaml_input:  Path to ARK YAML file(s), directory, or multi-doc YAML
    :param source_file: Path to Python file containing tools and graph_initializer
    :param project:     MLRun project (name, object, or None for current project)
    """

    def __init__(
        self,
        yaml_input,
        source_file: str,
        project=None,
    ):
        self.yaml_input = yaml_input
        self.source_file = source_file
        self.serving_function = None
        self._mock_server = None
        self._deployed = False

        if project is None:
            self._project = mlrun.get_current_project()
        elif isinstance(project, str):
            self._project = mlrun.get_or_create_project(project)
        else:
            self._project = project

    def build(
        self,
        name="declarative-agent",
        image=None,
        requirements=None,
        with_session=False,
    ):
        """Build the MLRun serving function with the appropriate graph topology.

        :param name:         Function name
        :param image:        Docker image (defaults to mlrun/mlrun)
        :param requirements: Python requirements list
        :param with_session: If True, adds SessionLoader and HistorySaver
        :return: MLRun serving function
        """
        default_reqs = [
            "git+https://github.com/mlrun/mlrun@agentic-ai-declerative-team-drawing",
            "langchain",
            "langchain-openai",
            "langgraph",
        ]

        self.serving_function = self._project.set_function(
            name=name,
            func=self.source_file,
            kind="serving",
            image=image or "mlrun/mlrun",
            requirements=requirements or default_reqs,
        )

        self.serving_function.spec.graph_initializer = "graph_initializer"

        # Load and classify YAMLs to decide graph topology
        docs = DeclarativeRunner._load_yamls(self.yaml_input)
        primary, members = DeclarativeRunner._classify_yamls(docs)

        root = self.serving_function.set_topology("flow", engine="async")

        if primary.get("kind") == "Team":
            # Team case: check strategy to decide topology
            strategy = primary.get("spec", {}).get("strategy", "sequential")

            if strategy == "sequential":
                # Build a linear chain of agent steps (no router)
                self._build_sequential_chain(
                    root, primary, members, with_session=with_session
                )
            else:
                # Use router with LangGraph for complex teams
                self._build_team_router(
                    root, primary, members, with_session=with_session
                )
        else:
            # Single agent: use flat DeclarativeRunner as before
            self._build_agent_graph(root, with_session=with_session)

        self.serving_function.spec.graph.plot(rankdir="LR")
        return self.serving_function

    def _build_sequential_chain(self, root, team_config, members, with_session=False):
        """Build a linear chain of DeclarativeAgent steps (no router, no LangGraph).

        Each agent is a real step that executes independently. Agents with the same
        team_label are visually grouped in the graph plot.

        Flow: SessionLoader → agent1 → agent2 → agent3 → HistorySaver
        """
        member_refs = team_config.get("spec", {}).get("members", [])
        team_name = team_config.get("metadata", {}).get("name", "team")

        # Start with session loader if needed
        if with_session:
            current = root.to(SessionLoader(name="session-loader"))
        else:
            current = root

        # Build chain of agent steps with team_label for visual grouping
        for ref in member_refs:
            member_name = ref.get("name") if isinstance(ref, dict) else ref
            if member_name in members:
                agent_config = members[member_name]

                # Create DeclarativeAgent instance and add as step
                agent = DeclarativeAgent(name=member_name, agent_config=agent_config)
                agent_step = current.to(agent)

                # Add custom attribute for visual grouping (must be set AFTER .to())
                agent_step.team_label = team_name
                current = agent_step

        # End with history saver if needed
        if with_session:
            current.to(HistorySaver(name="history-saver")).respond()
        else:
            current.respond()

    def _build_team_router(self, root, team_config, members, with_session=False):
        """Build a serving graph with DeclarativeTeamRouter for complex teams.

        Uses LangGraph for orchestration (round-robin, selector, graph strategies).
        Each member agent appears as a child node of the router.
        """
        member_refs = team_config.get("spec", {}).get("members", [])

        # Create DeclarativeTeamRouter instance
        router = DeclarativeTeamRouter(name="team-router", team_config=team_config)

        # Pre-compute graph structure for visualization
        structure = router.get_internal_graph_structure()

        if with_session:
            session_loader = SessionLoader(name="session-loader")
            history_saver = HistorySaver(name="history-saver")
            router_step = root.to(session_loader).to(router)
        else:
            router_step = root.to(router)

        # Inject pre-computed structure into step for plotting
        router_step.class_args = router_step.class_args or {}
        router_step.class_args["_graph_structure"] = structure

        # Add routes for each member agent (for visualization + initialization)
        for ref in member_refs:
            member_name = ref.get("name") if isinstance(ref, dict) else ref
            if member_name in members:
                router_step.add_route(
                    member_name,
                    class_name=DeclarativeAgent,
                    agent_config=members[member_name],
                )

        if with_session:
            router_step.to(history_saver).respond()
        else:
            router_step.respond()

    def _build_agent_graph(self, root, with_session=False):
        """Build a serving graph with a flat DeclarativeRunner for a single agent."""
        runner_step = DeclarativeRunner(
            yaml_input=self.yaml_input, name="declarative-runner"
        )

        if with_session:
            session_loader = SessionLoader(name="session-loader")
            history_saver = HistorySaver(name="history-saver")
            root.to(session_loader).to(runner_step).to(history_saver).respond()
        else:
            root.to(runner_step).respond()

    def create_mock(self):
        """Create a mock server for local testing.

        :return: Mock server instance
        """
        if self.serving_function is None:
            raise RuntimeError("Call build() first")
        self._mock_server = self.serving_function.to_mock_server()
        return self._mock_server

    def deploy(self, **kwargs):
        """Deploy the serving function to the cluster.

        :param kwargs: Additional arguments passed to function.deploy()
        :return: Deployment URL
        """
        if self.serving_function is None:
            raise RuntimeError("Call build() first")
        url = self.serving_function.deploy(**kwargs)
        self._deployed = True
        return url

    def infer(self, query: str):
        """Run inference with the deployed or mock server.

        :param query: User query string
        :return: Answer string
        """
        event = {"query": query}
        if self._deployed:
            resp = self.serving_function.invoke("", body=event)
        elif self._mock_server:
            resp = self._mock_server.test("", body=event)
        else:
            raise RuntimeError("Call create_mock() or deploy() first")

        if hasattr(resp, "results"):
            return resp.results.get("answer", str(resp))
        return resp.get("answer", str(resp)) if isinstance(resp, dict) else str(resp)
