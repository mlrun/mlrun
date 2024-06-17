import os
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

def init_context(context):

    model_id = os.environ['MODEL_ID']
    cache_dir = os.environ['CACHE_DIR']
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    lm_model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

    # Create chroma client
    chroma_client = chromadb.PersistentClient(path=cache_dir)

    setattr(context.user_data, 'collection', chroma_client.get_collection("my_news"))
    
    pipe = pipeline(
        "text-generation",
        model=lm_model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        device_map="auto",
        )
    
    setattr(context.user_data, 'pipe', pipe)
    
def handler(context, event):
    
    question_json = json.loads(event.body)
    question = question_json['question']
    topic = question_json['topic']
    
    results = context.user_data.collection.query(query_texts=[topic], n_results=10)
    
    q_context = " ".join([f"#{str(i)}" for i in results["documents"][0]])
    prompt_template = f"Relevant context: {q_context}\n\n The user's question: {question}"
    
    lm_response = context.user_data.pipe(prompt_template)
    print(lm_response[0]["generated_text"])

    return lm_response[0]["generated_text"]