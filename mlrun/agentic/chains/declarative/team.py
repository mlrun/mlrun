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
from mlrun.agentic.chains.base import ChainRunner
from mlrun.agentic.chains.declarative.agent import DeclarativeAgent
from mlrun.agentic.chains.declarative.strategies import (
    build_graph_strategy_graph,
    build_round_robin_graph,
    build_selector_graph,
    build_sequential_graph,
)
from mlrun.agentic.schemas import WorkflowEvent

logger = mlrun.utils.logger

_STRATEGY_BUILDERS = {
    "sequential": "sequential",
    "round-robin": "round_robin",
    "roundRobin": "round_robin",
    "selector": "selector",
    "graph": "graph",
}


class DeclarativeTeam(ChainRunner):
    """A ChainRunner that orchestrates multiple DeclarativeAgents using a strategy.

    Supports sequential, round-robin, selector (supervisor), and graph strategies
    as defined in the ARK Team YAML spec.

    :param team_config:     Pre-parsed YAML dict for the team (kind: Team).
    :param members_by_name: Dict mapping agent name -> agent config dict.
    """

    def __init__(self, team_config: dict, members_by_name: dict[str, dict], **kwargs):
        super().__init__(**kwargs)
        self.team_config = team_config
        self.members_by_name = members_by_name
        self.agents = {}
        self.compiled_graph = None

    def post_init(self, mode="sync", context=None, namespace=None, **kwargs):
        """Build DeclarativeAgents for each member and compile the strategy graph.

        Parses the ARK Team YAML spec where:
        - ``spec.strategy`` is a plain string (sequential, round-robin, selector, graph)
        - ``spec.maxTurns`` is at spec level
        - ``spec.selector`` is at spec level (for selector/hybrid strategies)
        - ``spec.graph`` is at spec level (for graph/hybrid strategies)
        """
        spec = self.team_config.get("spec", {})

        # ARK schema: strategy is a plain string at spec level
        strategy_type = spec.get("strategy", "sequential")
        max_turns = spec.get("maxTurns", 10)
        selector_spec = spec.get("selector", {})
        graph_spec = spec.get("graph", {})

        # Resolve ordered member list from team spec
        member_refs = spec.get("members", [])
        member_names = []
        for ref in member_refs:
            name = ref.get("name") if isinstance(ref, dict) else ref
            member_names.append(name)

        # Build agents for each member
        for name in member_names:
            if name not in self.members_by_name:
                raise ValueError(
                    f"Team references member '{name}' but no Agent config found"
                )
            agent = DeclarativeAgent(agent_config=self.members_by_name[name])
            agent.post_init(mode=mode, context=context, namespace=namespace, **kwargs)
            self.agents[name] = agent

        # Normalize strategy name and build the appropriate graph
        normalized = _STRATEGY_BUILDERS.get(strategy_type, strategy_type)

        if normalized == "sequential":
            self.compiled_graph = build_sequential_graph(member_names, self.agents)

        elif normalized == "round_robin":
            self.compiled_graph = build_round_robin_graph(
                member_names, self.agents, max_turns=max_turns
            )

        elif normalized == "selector":
            # Selector strategy; if graph edges also present, use hybrid graph+selector
            if graph_spec.get("edges"):
                self.compiled_graph = build_graph_strategy_graph(
                    member_names,
                    self.agents,
                    graph_spec=graph_spec,
                    selector_spec=selector_spec,
                    max_turns=max_turns,
                )
            else:
                self.compiled_graph = build_selector_graph(
                    member_names,
                    self.agents,
                    selector_spec=selector_spec,
                    max_turns=max_turns,
                )

        elif normalized == "graph":
            self.compiled_graph = build_graph_strategy_graph(
                member_names,
                self.agents,
                graph_spec=graph_spec,
                selector_spec=selector_spec or None,
                max_turns=max_turns,
            )

        else:
            raise ValueError(f"Unknown team strategy: {strategy_type}")

        logger.info(
            "Team graph compiled",
            strategy=strategy_type,
            members=member_names,
        )

    def _run(self, event: WorkflowEvent):
        """Bridge WorkflowEvent to TeamState, invoke the compiled graph, extract answer."""
        query = event.query.content if hasattr(event.query, "content") else event.query

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
