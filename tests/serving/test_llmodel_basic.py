# Copyright 2026 Iguazio
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

import mlrun
from mlrun.serving.states import LLModel


def test_enrich_prompt_batch():
    """Test that enrich_prompt can handle both single dict and list of dicts (batch)."""
    model = LLModel(name="test_model", input_path="data")

    # Create LLMPromptArtifact with template
    project = mlrun.new_project("test-enrich-prompt-batch", save=False)
    prompt_artifact = project.log_llm_prompt(
        key="test-prompt",
        prompt_template=[
            {
                "role": "user",
                "content": "{question}. Explain {depth_level} as a {persona} in {tone} style.",
            }
        ],
    )

    # Test 1: Single event (dict)
    single_event = {
        "data": {
            "question": "What is the capital of France",
            "depth_level": "basic",
            "persona": "child",
            "tone": "fun",
        }
    }

    enriched_messages, invocation_config = model.enrich_prompt(
        body=single_event, origin_name="test", llm_prompt_artifact=prompt_artifact
    )

    assert enriched_messages == [
        {
            "role": "user",
            "content": "What is the capital of France. Explain basic as a child in fun style.",
        }
    ]

    # Test 2: Batch events (list of dicts)
    batch_events = {
        "data": [
            {
                "question": "What color is the sky",
                "depth_level": "basic",
                "persona": "child",
                "tone": "fun",
            },
            {
                "question": "How does gravity work",
                "depth_level": "advanced",
                "persona": "scientist",
                "tone": "formal",
            },
            {
                "question": "Why do birds fly",
                "depth_level": "intermediate",
                "persona": "student",
                "tone": "casual",
            },
        ]
    }

    enriched_messages_list, invocation_config = model.enrich_prompt(
        body=batch_events, origin_name="test", llm_prompt_artifact=prompt_artifact
    )

    assert enriched_messages_list == [
        [
            {
                "role": "user",
                "content": "What color is the sky. Explain basic as a child in fun style.",
            }
        ],
        [
            {
                "role": "user",
                "content": "How does gravity work. Explain advanced as a scientist in formal style.",
            }
        ],
        [
            {
                "role": "user",
                "content": "Why do birds fly. Explain intermediate as a student in casual style.",
            }
        ],
    ]

    # Test 3: Batch with no template
    model_no_template = LLModel(name="test_model_no_template", input_path="data")

    batch_events_no_template = {
        "data": [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "World"}]},
        ]
    }

    enriched_messages_list, invocation_config = model_no_template.enrich_prompt(
        body=batch_events_no_template, origin_name="test", llm_prompt_artifact=None
    )

    assert enriched_messages_list == [
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "World"}],
    ]
