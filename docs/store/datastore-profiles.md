(datastore-profies)=
# Datastore profiles

Data store profiles serve multiple purposes in MLRun:

GenAI: OpenAI and HF
Model monitoring: TimescaleDB
Queues: Kafka, RabbitMQ
Storage providers: See  {ref}`data-stores`

## GenAI datastore profiles

GenAI datastore profiles define credentials and environment variables. 

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
```
See
- {py:class}`mlrun.datastore.datastore_profile.OpenAIProfile`
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
```

See
- {py:class}`mlrun.datastore.datastore_profile.HuggingFaceProfile`
- [Integrating a Hugging Face image classification model with MLRun](../genai/deployment/hf-model-image-classification.ipynb)

## Model monitoring datastore profiles

Model monitoring uses a streaming platform and a TSDB platform. It supports
Kafka and V3IO as streaming platforms, and TimescaleDB (PostgreSQL) and V3IO as TSDB platforms.

TimescaleDB (PostgreSQL) and Kafka are part of the default CE installations. The default confgurations are:
```
# Create and register TSDB profile
tsdb_profile = DatastoreProfileTDEngine(
    name=tsdb_profile_name,
    user="root",
    password="taosdata",
    host=f"tdengine-tsdb",
    port="6041",
)
project.register_datastore_profile(tsdb_profile)

# Create and register stream profile
stream_profile = DatastoreProfileKafkaSource(
    name=stream_profile_name,
    brokers=f"kafka-stream:9092",
    topics=[],
)
```

The V3IO configurations are:
```
tsdb_profile = DatastoreProfileV3io(
    name="my-v3io-tsdb",
)

stream_profile = DatastoreProfileV3io(
    name="my-v3io-stream",
    v3io_access_key=mlrun.mlconf.get_v3io_access_key(),
)
```
See {py:class}`mlrun.projects.MlrunProject.set_model_monitoring_credentials`.

## Queue datastore profiles

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
### Kafka

```
profile = DatastoreProfileKafkaStream(
    name="profile-name", brokers="localhost", topic="topic_name"
)
# add this profile to a function with:
target = KafkaSource(path="ds://profile-name")
```
See full details in {py:class}`~mlrun.datastore.datastore_profile.DatastoreProfileKafkaStream`.