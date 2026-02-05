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

from mlrun.agentic.chains.base import ChainRunner, HistorySaver, SessionLoader
from mlrun.agentic.chains.communicator import Communicator
from mlrun.agentic.chains.hallucination_guardrail import HallucinationGuardrail
from mlrun.agentic.chains.intent_choice import IntentChoice
from mlrun.agentic.chains.intent_classifier import IntentClassifier
from mlrun.agentic.chains.language_guardrail import LanguageGuardrail
from mlrun.agentic.chains.refine import (
    CONVERSATION_CONTEXT_REFINER_PROMPT,
    RefineQuery,
    get_refine_chain,
)
from mlrun.agentic.chains.retrieval import (
    DocumentCallbackHandler,
    DocumentRetriever,
    MultiRetriever,
    fix_milvus_filter_arg,
    get_retriever_from_config,
)

# Optional imports
try:
    from mlrun.agentic.chains.a2a_client import A2AClient
except ImportError:
    A2AClient = None

try:
    from mlrun.agentic.chains.sentiment_analysis import SentimentAnalysisStep
except ImportError:
    SentimentAnalysisStep = None

try:
    from mlrun.agentic.chains.declarative_agent import DeclarativeAgent
except ImportError:
    DeclarativeAgent = None
