import json
import os

import chromadb
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
    set_seed,
)

PROMPT_TEMPLATE = """The instruction below describes a task. Write a response that appropriately completes the request.

### Instruction:
User question:
{question}

Context:
{context}

### Response:
"""


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: list):
        StoppingCriteria.__init__(self)
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # print(input_ids)
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def init_context(context):
    model_id = os.environ["MODEL_ID"]
    cache_dir = os.environ["CACHE_DIR"]

    # Initialize HF models
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    lm_model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

    # Create chroma client and collection
    chroma_client = chromadb.PersistentClient(path=cache_dir)
    setattr(context.user_data, "collection", chroma_client.get_collection("my_news"))

    # Model will stop generating after #, ##, or ###
    # This speeds up inference and prevents garbage at the end
    stop_token_ids = tokenizer.convert_tokens_to_ids(["#", "##", "###"])
    stopping_criteria = StoppingCriteriaList(
        [StopOnTokens(stop_token_ids=stop_token_ids)]
    )

    # HF text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=lm_model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        device_map="auto",
        stopping_criteria=stopping_criteria,
    )

    setattr(context.user_data, "pipe", pipe)


def handler(context, event):
    # Unpack payload
    question_json = json.loads(event.body)
    question = question_json["question"]
    topic = question_json["topic"]

    # Optional seed for deterministic responses
    seed = question_json.get("seed", None)
    if seed:
        set_seed(int(seed))

    # Query vector store
    results = context.user_data.collection.query(
        query_texts=[question], n_results=3, where={"topic": {"$eq": topic.upper()}}
    )
    sources = list({r["link"] for r in results["metadatas"][0]})

    # Construct prompt
    q_context = "\n\n".join(results["documents"][0])
    prompt = PROMPT_TEMPLATE.format(question=question, context=q_context)

    # Generate result
    resp = context.user_data.pipe(prompt)
    generated = resp[0]["generated_text"][len(prompt) :].split("#")[0]

    return {"sources": sources, "prompt": prompt, "response": generated}
