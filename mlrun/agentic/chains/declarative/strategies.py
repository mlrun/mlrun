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

import operator
from typing import Annotated

from langchain.agents import create_agent
from langgraph.graph import END, StateGraph

import mlrun.utils

logger = mlrun.utils.logger


class TeamState(dict):
    """State shared across a team's LangGraph execution.

    Fields:
        messages: Accumulated conversation messages (append-only via operator.add).
        current_turn: Counter for round-robin / selector turn tracking.
        max_turns: Maximum allowed turns before termination.
        last_agent: Name of the most recently invoked agent.
        final_answer: The final output to return.
    """

    __annotations__ = {
        "messages": Annotated[list, operator.add],
        "current_turn": int,
        "max_turns": int,
        "last_agent": str,
        "final_answer": str,
    }


def _make_agent_node(name: str, agent):
    """Create a LangGraph node function that wraps a DeclarativeAgent.

    The node invokes the agent with the accumulated messages, appends the
    agent's response to the message list, and updates the turn counter.

    :param name:  Agent name (used as message role identifier).
    :param agent: A DeclarativeAgent instance (must have invoke_with_messages).
    :return: A callable suitable for StateGraph.add_node.
    """

    def node_fn(state: dict) -> dict:
        messages = state.get("messages", [])
        result = agent.invoke_with_messages(messages)
        answer = result.get("answer", "")
        return {
            "messages": [{"role": name, "content": answer}],
            "current_turn": state.get("current_turn", 0) + 1,
            "last_agent": name,
            "final_answer": answer,
        }

    node_fn.__name__ = f"agent_node_{name}"
    return node_fn


def build_sequential_graph(member_names: list[str], agents: dict) -> StateGraph:
    """Build a sequential execution graph: A -> B -> C -> END.

    :param member_names: Ordered list of agent names.
    :param agents:       Dict mapping name -> DeclarativeAgent.
    :return: Compiled StateGraph.
    """
    graph = StateGraph(TeamState)

    for name in member_names:
        graph.add_node(name, _make_agent_node(name, agents[name]))

    graph.set_entry_point(member_names[0])

    for i in range(len(member_names) - 1):
        graph.add_edge(member_names[i], member_names[i + 1])

    graph.add_edge(member_names[-1], END)

    return graph.compile()


def build_round_robin_graph(
    member_names: list[str], agents: dict, max_turns: int = 10
) -> StateGraph:
    """Build a round-robin cycling graph: Router dispatches to agents in order.

    Agents cycle in round-robin order until current_turn exceeds max_turns.

    :param member_names: Ordered list of agent names.
    :param agents:       Dict mapping name -> DeclarativeAgent.
    :param max_turns:    Maximum number of agent invocations.
    :return: Compiled StateGraph.
    """
    graph = StateGraph(TeamState)

    def router(state: dict) -> str:
        turn = state.get("current_turn", 0)
        if turn >= max_turns:
            return END
        idx = turn % len(member_names)
        return member_names[idx]

    graph.add_node("router", lambda state: state)

    for name in member_names:
        graph.add_node(name, _make_agent_node(name, agents[name]))

    graph.set_entry_point("router")

    destinations = {name: name for name in member_names}
    destinations[END] = END
    graph.add_conditional_edges("router", router, destinations)

    for name in member_names:
        graph.add_edge(name, "router")

    return graph.compile()


def build_selector_graph(
    member_names: list[str],
    agents: dict,
    selector_spec: dict,
    max_turns: int = 10,
) -> StateGraph:
    """Build a supervisor/selector graph where an LLM picks the next agent.

    The selector node uses the provided prompt template and LLM to decide
    which agent should act next. Prevents the same agent from being selected
    twice in a row. Falls back to the first valid member on invalid selection.

    :param member_names:  List of agent names.
    :param agents:        Dict mapping name -> DeclarativeAgent.
    :param selector_spec: ARK selector dict with 'agent' (coordinator name) and
                          'selectorPrompt' (Go template with {{.Roles}}, {{.Participants}},
                          {{.History}} placeholders).
    :param max_turns:     Maximum number of agent invocations.
    :return: Compiled StateGraph.
    """
    selector_prompt_template = selector_spec.get("selectorPrompt", "")
    # The selector agent reference; its modelRef is used for the routing LLM
    selector_agent_name = selector_spec.get("agent", "")
    if selector_agent_name and selector_agent_name in agents:
        selector_model = (
            agents[selector_agent_name]
            .agent_config.get("spec", {})
            .get("modelRef", {})
            .get("name", "gpt-4")
        )
    else:
        selector_model = "gpt-4"

    selector_agent = create_agent(
        model=selector_model,
        tools=[],
        system_prompt=(
            "You are a team coordinator. Given the conversation history and team member "
            "roles, select the next agent to act. Respond with ONLY the agent name."
        ),
    )

    roles_text = "\n".join(
        f"- {name}: {agents[name].agent_config.get('spec', {}).get('prompt', 'No description')[:200]}"
        for name in member_names
    )
    participants_text = ", ".join(member_names)

    graph = StateGraph(TeamState)

    def selector_node(state: dict) -> dict:
        turn = state.get("current_turn", 0)
        if turn >= max_turns:
            return state

        messages = state.get("messages", [])
        last_agent = state.get("last_agent", "")

        history_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages[-10:]
        )

        prompt = selector_prompt_template
        prompt = prompt.replace("{{.Roles}}", roles_text)
        prompt = prompt.replace("{{.Participants}}", participants_text)
        prompt = prompt.replace("{{.History}}", history_text)

        result = selector_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        selector_messages = result.get("messages", [])
        if selector_messages:
            last_msg = selector_messages[-1]
            choice = (
                last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            ).strip()
        else:
            choice = ""

        # Prevent same agent twice in a row
        valid = [n for n in member_names if n != last_agent]
        if not valid:
            valid = member_names

        if choice not in valid:
            logger.warning(
                "Selector chose invalid agent, falling back",
                choice=choice,
                valid=valid,
            )
            choice = valid[0]

        return {
            "messages": [{"role": "selector", "content": f"Selected agent: {choice}"}],
            "last_agent": state.get("last_agent", ""),
            "current_turn": turn,
            "final_answer": state.get("final_answer", ""),
            "_next": choice,
        }

    def route_selector(state: dict) -> str:
        turn = state.get("current_turn", 0)
        if turn >= max_turns:
            return END
        return state.get("_next", member_names[0])

    graph.add_node("selector", selector_node)

    for name in member_names:
        graph.add_node(name, _make_agent_node(name, agents[name]))

    graph.set_entry_point("selector")

    destinations = {name: name for name in member_names}
    destinations[END] = END
    graph.add_conditional_edges("selector", route_selector, destinations)

    for name in member_names:
        graph.add_edge(name, "selector")

    return graph.compile()


def build_graph_strategy_graph(
    member_names: list[str],
    agents: dict,
    graph_spec: dict,
    selector_spec: dict | None = None,
    max_turns: int = 10,
) -> StateGraph:
    """Build a custom DAG from YAML graph edges, optionally with a selector for branching.

    Nodes with no outgoing edges route to END. If selector_spec is provided (hybrid mode),
    nodes with multiple outgoing edges use a selector LLM to pick the next node; nodes with
    a single outgoing edge skip the selector and transition directly.

    :param member_names:  List of agent names.
    :param agents:        Dict mapping name -> DeclarativeAgent.
    :param graph_spec:    Dict with 'edges' list of {'from': str, 'to': str}.
    :param selector_spec: Optional dict with selector prompt/model for hybrid mode.
    :param max_turns:     Maximum turns (used in hybrid mode).
    :return: Compiled StateGraph.
    """
    edges = graph_spec.get("edges", [])

    # Build adjacency map: source -> [targets]
    adjacency = {}
    all_targets = set()
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        adjacency.setdefault(src, []).append(dst)
        all_targets.add(dst)

    graph = StateGraph(TeamState)

    for name in member_names:
        graph.add_node(name, _make_agent_node(name, agents[name]))

    # Find entry point: first member, or first node with no incoming edges
    nodes_with_incoming = all_targets
    entry_candidates = [n for n in member_names if n not in nodes_with_incoming]
    entry_point = entry_candidates[0] if entry_candidates else member_names[0]
    graph.set_entry_point(entry_point)

    # Build selector agent if hybrid mode
    hybrid_selector = None
    if selector_spec:
        hybrid_model = selector_spec.get("modelRef", {}).get("name", "gpt-4")
        hybrid_selector = create_agent(
            model=hybrid_model,
            tools=[],
            system_prompt=(
                "You are a team coordinator. Given the conversation and available next agents, "
                "select the next agent. Respond with ONLY the agent name."
            ),
        )

    for name in member_names:
        outgoing = adjacency.get(name, [])

        if not outgoing:
            # Terminal node -> END
            graph.add_edge(name, END)

        elif len(outgoing) == 1:
            # Single outgoing edge -> direct transition
            graph.add_edge(name, outgoing[0])

        elif hybrid_selector:
            # Multiple outgoing edges with selector
            _add_hybrid_selector_edges(
                graph, name, outgoing, agents, hybrid_selector, max_turns
            )
        else:
            # Multiple outgoing edges without selector -> take first
            logger.warning(
                "Multiple outgoing edges without selector, using first edge",
                node=name,
                edges=outgoing,
            )
            graph.add_edge(name, outgoing[0])

    return graph.compile()


def _add_hybrid_selector_edges(
    graph: StateGraph,
    node_name: str,
    outgoing: list[str],
    agents: dict,
    hybrid_selector,
    max_turns: int,
):
    """Add conditional edges from a node using a selector LLM to choose among outgoing targets.

    :param graph:           The StateGraph being constructed.
    :param node_name:       Name of the source node.
    :param outgoing:        List of valid target node names.
    :param agents:          Dict mapping name -> DeclarativeAgent.
    :param hybrid_selector: LLM agent for making routing decisions.
    :param max_turns:       Max turns for termination.
    """
    choices_text = ", ".join(outgoing)

    def hybrid_route(state: dict) -> str:
        turn = state.get("current_turn", 0)
        if turn >= max_turns:
            return END

        messages = state.get("messages", [])
        history_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages[-10:]
        )

        prompt = (
            f"Based on the conversation so far, choose the next agent from: {choices_text}\n\n"
            f"Recent conversation:\n{history_text}\n\nNext agent:"
        )

        result = hybrid_selector.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        selector_messages = result.get("messages", [])
        if selector_messages:
            last_msg = selector_messages[-1]
            choice = (
                last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            ).strip()
        else:
            choice = ""

        if choice not in outgoing:
            logger.warning(
                "Hybrid selector chose invalid target, falling back",
                choice=choice,
                valid=outgoing,
            )
            choice = outgoing[0]
        return choice

    destinations = {name: name for name in outgoing}
    destinations[END] = END
    graph.add_conditional_edges(node_name, hybrid_route, destinations)
