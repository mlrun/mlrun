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

import mlrun.utils
from mlrun.agentic.chains.declarative.strategies import (
    build_graph_strategy_graph,
    build_round_robin_graph,
    build_selector_graph,
    build_sequential_graph,
)
from mlrun.agentic.schemas import WorkflowEvent
from mlrun.serving.routers import BaseModelRouter

logger = mlrun.utils.logger

_STRATEGY_BUILDERS = {
    "sequential": "sequential",
    "round-robin": "round_robin",
    "roundRobin": "round_robin",
    "selector": "selector",
    "graph": "graph",
}


class DeclarativeTeamRouter(BaseModelRouter):
    """A serving router that orchestrates a team of DeclarativeAgents via LangGraph.

    Routes are added for graph visualization — each member agent appears as a child
    node in the serving graph. At runtime, ``do_event`` runs the compiled LangGraph
    workflow instead of routing to individual models.

    :param context:     For internal use (passed during init).
    :param name:        Step name.
    :param routes:      For internal use (routes passed during init).
    :param team_config: Pre-parsed YAML dict for the team (kind: Team).
    """

    def __init__(self, context=None, name=None, routes=None, team_config=None, **kwargs):
        super().__init__(context=context, name=name, routes=routes, **kwargs)
        self.team_config = team_config or {}
        self.compiled_graph = None

    def post_init(self, mode="sync", **kwargs):
        """Build agents from initialized routes and compile the strategy graph.

        Called after all route ``init_object`` calls have completed, so each
        route's ``_object`` is a fully initialized ``DeclarativeAgent``.
        """
        # Extract already-initialized DeclarativeAgent instances from routes
        agents = {}
        for route_name, route_step in self.routes.items():
            agents[route_name] = route_step._object

        spec = self.team_config.get("spec", {})
        strategy_type = spec.get("strategy", "sequential")
        max_turns = spec.get("maxTurns", 10)
        selector_spec = spec.get("selector", {})
        graph_spec = spec.get("graph", {})

        # Resolve ordered member list from team spec
        member_refs = spec.get("members", [])
        member_names = [
            ref.get("name") if isinstance(ref, dict) else ref for ref in member_refs
        ]

        # Normalize strategy name and build the appropriate graph
        normalized = _STRATEGY_BUILDERS.get(strategy_type, strategy_type)

        if normalized == "sequential":
            self.compiled_graph = build_sequential_graph(member_names, agents)

        elif normalized == "round_robin":
            self.compiled_graph = build_round_robin_graph(
                member_names, agents, max_turns=max_turns
            )

        elif normalized == "selector":
            if graph_spec.get("edges"):
                self.compiled_graph = build_graph_strategy_graph(
                    member_names,
                    agents,
                    graph_spec=graph_spec,
                    selector_spec=selector_spec,
                    max_turns=max_turns,
                )
            else:
                self.compiled_graph = build_selector_graph(
                    member_names,
                    agents,
                    selector_spec=selector_spec,
                    max_turns=max_turns,
                )

        elif normalized == "graph":
            self.compiled_graph = build_graph_strategy_graph(
                member_names,
                agents,
                graph_spec=graph_spec,
                selector_spec=selector_spec or None,
                max_turns=max_turns,
            )

        else:
            raise ValueError(f"Unknown team strategy: {strategy_type}")

        logger.info(
            "DeclarativeTeamRouter initialized",
            strategy=strategy_type,
            members=member_names,
        )

    def do_event(self, event, *args, **kwargs):
        """Handle incoming events by running the LangGraph workflow.

        Supports both ``WorkflowEvent`` bodies (when used with SessionLoader)
        and plain dict bodies (direct invocation).
        """
        body = event.body

        if isinstance(body, WorkflowEvent):
            query = body.query.content if hasattr(body.query, "content") else body.query
            result = self._invoke_graph(query)
            body.results["answer"] = result["answer"]
            body.query = result["answer"]
        elif isinstance(body, dict):
            query = body.get("query", "")
            result = self._invoke_graph(query)
            event.body = result
        else:
            query = str(body)
            result = self._invoke_graph(query)
            event.body = result

        return event

    def _invoke_graph(self, query):
        """Run the compiled LangGraph workflow and extract the answer.

        :param query: User query string.
        :return: Dict with 'answer' key.
        """
        max_turns = self.team_config.get("spec", {}).get("maxTurns", 10)
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "current_turn": 0,
            "max_turns": max_turns,
            "last_agent": "",
            "final_answer": "",
        }

        result_state = self.compiled_graph.invoke(initial_state)

        answer = result_state.get("final_answer", "")
        if not answer:
            messages = result_state.get("messages", [])
            if messages:
                last = messages[-1]
                answer = (
                    last.get("content", "") if isinstance(last, dict) else str(last)
                )

        return {"answer": answer}
