(profies)=
# Profiles

Profiles are containers for credentials for a remote service. See:
- Datastore: see {ref}`datastore`
- Config, general-purpose, used in vectorDBs. See [Milvus configuration](../genai/data-mgmt/vector-databases.md#milvus-configuration)

**In this section**
- [Provider profiles](#model-provider-profiles)
- [Source and target profiles](#source-and-target-profiles)

Profiles are also used when configuring model monitoring. See [Configuring TimescaleDB and Kafka for model monitoring](../install-mlrun-ce/mlrun-ce-development-notes.md#configuring-timescaledb-and-kafka-for-model-monitoring).

## Model provider profiles

Model provider profiles define credentials and environment variables for remote-model providers (for predictions). 

### OpenAI profile
```python
open_ai_profile = OpenAIProfile(
    name="openai_profile",
    api_key=os.environ.get("OPENAI_API_KEY"),
    organization=os.environ.get("OPENAI_ORG_ID"),
    project=os.environ.get("OPENAI_PROJECT_ID"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
    timeout=os.environ.get("OPENAI_TIMEOUT"),
    max_retries=os.environ.get("OPENAI_MAX_RETRIES"),
)

project.register_datastore_profile(open_ai_profile)
model_url = f"ds://openai_profile/model-name"
```
See also:
- {py:class}`~mlrun.datastore.datastore_profile.OpenAIProfile`
- [Integrating an OpenAI LLM with MLRun](../genai/deployment/openai-model.ipynb)

### Hugging Face
```python
profile = HuggingFaceProfile(
    name=profile_name,
    task="image-classification",
    token=os.environ.get("HF_TOKEN"),
    device=os.environ.get("HF_DEVICE"),
    device_map=os.environ.get("HF_DEVICE_MAP"),
    trust_remote_code=os.environ.get("HF_TRUST_REMOTE_CODE"),
)

# Register the profile with the project
project.register_datastore_profile(profile)
```

See also:
- {py:class}`~mlrun.datastore.datastore_profile.HuggingFaceProfile`
- {ref}`hf-model-batch-serving-graph`

## Source and target profiles

### RabbitMQ
```
profile = DatastoreProfileRabbitMQ(
    name="my-profile",
    url="amqp://host:5672",
    exchange_name="my-exchange",
    queue_name="my-queue",
    num_workers=4,
)

# add this profile to a function with:
function.add_rabbitmq_trigger(url="ds://my-profile")
```

See also
- {py:class}`~mlrun.datastore.datastore_profile.DatastoreProfileRabbitMQ`
- {py:class}`~mlrun.runtimes.RemoteRuntime.add_rabbitmq_trigger`

### Kafka

```
profile = DatastoreProfileKafkaStream(
    name="profile-name", brokers="localhost", topic="topic_name"
)
# add this profile to a function with:
target = KafkaSource(path="ds://profile-name")
```
See also
- {py:class}`~mlrun.datastore.datastore_profile.DatastoreProfileKafkaStream`
- {py:class}`~mlrun.datastore.datastore_profile.DatastoreProfileKafkaSource`
- {py:class}`~mlrun.datastore.datastore_profile.DatastoreProfileKafkaTarget`