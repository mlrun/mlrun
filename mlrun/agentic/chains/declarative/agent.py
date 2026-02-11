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

import sys

from langchain.agents import create_agent

from mlrun.agentic.chains.base import ChainRunner
from mlrun.agentic.schemas import WorkflowEvent


class DeclarativeAgent(ChainRunner):
    """A ChainRunner step that creates a LangChain agent from a parsed YAML config dict.

    :param agent_config: Pre-parsed YAML dict for the agent (must have kind: Agent).
    """

    def __init__(self, agent_config: dict, **kwargs):
        super().__init__(**kwargs)
        self.agent_config = agent_config
        self.agent = None

    def post_init(self, mode="sync", context=None, namespace=None, **kwargs):
        """Parse config and create the LangChain agent."""
        spec = self.agent_config.get("spec", {})
        prompt = spec.get("prompt", "")

        model_ref = spec.get("modelRef", {})
        model_name = model_ref.get("name", "gpt-4")

        tools = self._resolve_tools(spec.get("tools", []))

        self.agent = create_agent(
            model=model_name,
            tools=tools,
            system_prompt=prompt,
        )

    def _resolve_tools(self, tool_refs: list) -> list:
        """Resolve tool references by searching sys.modules for matching names.

        :param tool_refs: List of tool references (dicts with 'name' or plain strings).
        :return: List of resolved tool objects.
        """
        tool_names = []
        for tool_ref in tool_refs:
            name = tool_ref.get("name") if isinstance(tool_ref, dict) else tool_ref
            tool_names.append(name)

        tools = []
        for _module_name, module in sys.modules.items():
            if module is None:
                continue
            for tool_name in tool_names:
                if hasattr(module, tool_name):
                    tool = getattr(module, tool_name)
                    if hasattr(tool, "invoke") and tool not in tools:
                        tools.append(tool)
        return tools

    def invoke_with_messages(self, messages: list[dict]) -> dict:
        """Invoke the agent with a full message history.

        Used by team strategies to pass accumulated conversation context.

        :param messages: List of message dicts with 'role' and 'content' keys.
        :return: Agent result dict.
        """
        result = self.agent.invoke({"messages": messages})
        messages_out = result.get("messages", [])
        if messages_out:
            last_msg = messages_out[-1]
            answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        else:
            answer = str(result)
        return {"answer": answer}

    def _run(self, event: WorkflowEvent):
        """Extract query from event, invoke agent, return answer."""
        query = event.query.content if hasattr(event.query, "content") else event.query
        return self.invoke_with_messages([{"role": "user", "content": query}])
