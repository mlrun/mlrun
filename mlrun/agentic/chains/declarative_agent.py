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

import yaml

from langchain.agents import create_agent

from mlrun.agentic.chains.base import ChainRunner
from mlrun.agentic.schemas import WorkflowEvent


class DeclarativeAgent(ChainRunner):
    """A ChainRunner step that creates a LangChain agent from YAML config."""

    def __init__(self, agent_yaml: str, **kwargs):
        super().__init__(**kwargs)
        self.agent_yaml = agent_yaml
        self.agent = None

    def post_init(self, mode="sync", context=None, namespace=None, **kwargs):
        """Parse YAML and create the LangChain agent."""
        with open(self.agent_yaml, "r") as f:
            data = yaml.safe_load(f)

        spec = data.get("spec", {})
        prompt = spec.get("prompt", "")

        model_ref = spec.get("modelRef", {})
        model_name = model_ref.get("name", "gpt-4")

        # Resolve tools by searching sys.modules
        import sys
        tool_names_to_find = []
        for tool_ref in spec.get("tools", []):
            tool_name = tool_ref.get("name") if isinstance(tool_ref, dict) else tool_ref
            tool_names_to_find.append(tool_name)

        tools = []
        for module_name, module in sys.modules.items():
            if module is None:
                continue
            for tool_name in tool_names_to_find:
                if hasattr(module, tool_name):
                    tool = getattr(module, tool_name)
                    # Check if it's a langchain tool (has .invoke method)
                    if hasattr(tool, "invoke") and tool not in tools:
                        tools.append(tool)

        self.agent = create_agent(
            model=model_name,
            tools=tools,
            system_prompt=prompt,
        )

    def _run(self, event: WorkflowEvent):
        query = event.query.content if hasattr(event.query, "content") else event.query
        result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        else:
            answer = str(result)
        return {"answer": answer}
