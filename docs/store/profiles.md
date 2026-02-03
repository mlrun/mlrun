(profies)=
# Profiles

Profiles are containers for credentials for a remote service, for example, storage, model or stream service. 

In this section:
- [Model provider profiles](#model-provider-profiles)
- [Source and target profiles](#source-and-target-profiles)

Storage provider datastore profiles are described in {ref}`datastore`.

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
- [Integrating a Hugging Face image classification model with MLRun](../genai/deployment/hf-model-image-classification.ipynb)

## Source and target profiles

Source and target profiles include model monitoring profiles and queue profiles
### Model monitoring datastore profiles
Model monitoring datastore profiles define the streaming and TSDB platforms required to run model monitoring.
MLRun supports Kafka and V3IO as streaming platforms, and TimescaleDB (PostgreSQL) and V3IO as TSDB platforms.

TimescaleDB (PostgreSQL) and Kafka are part of the default CE installations. The default confgurations are:
```
# Create and register TSDB profile
tsdb_profile = DatastoreProfilePostgreSQL(
    name="my-timescaledb",
    host="<timescaledb-server-ip-address>",
    port=5432,
    user="postgres",
    password="<timescaledb-password>",
    database="mlrun",

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
See also
- {py:class}`~mlrun.projects.MlrunProject.set_model_monitoring_credentials`.
- [Configuring TimescaleDB and Kafka for model monitoring](../install-mlrun-ce/mlrun-ce-development-notes.md#configuring-timescaledb-and-kafka-for-model-monitoring)

### Queue datastore profiles

#### RabbitMQ
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

#### Kafka

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