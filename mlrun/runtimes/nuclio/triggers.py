# Copyright 2024 Iguazio
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

from typing import NamedTuple, Optional
from urllib.parse import unquote, urlparse, urlunparse

from nuclio.triggers import NuclioTrigger

import mlrun.datastore.datastore_profile


class UrlCredentials(NamedTuple):
    """Parsed URL with extracted and decoded credentials."""

    url: str
    username: Optional[str]
    password: Optional[str]


def _first_not_none(*values):
    """Return the first non-None value, or None if all are None."""
    for v in values:
        if v is not None:
            return v
    return None


def extract_credentials_from_url(url: str) -> UrlCredentials:
    """
    Extract credentials from URL and return clean URL without embedded credentials.

    Credentials are URL-decoded to handle special characters (e.g., %40 -> @).

    :param url: URL that may contain embedded credentials (e.g., 'amqp://user:pass@host:port')
    :return: UrlCredentials with clean_url, decoded username, and decoded password
    """
    parsed = urlparse(url)

    if not parsed.username and not parsed.password:
        return UrlCredentials(url, None, None)

    # Reconstruct URL without credentials
    hostname = parsed.hostname or ""
    netloc = f"{hostname}:{parsed.port}" if parsed.port else hostname
    clean_url = urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )

    # Decode URL-encoded characters (e.g., %40 -> @, %20 -> space)
    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None

    return UrlCredentials(clean_url, username, password)


class RabbitMQTrigger(NuclioTrigger):
    """
    RabbitMQ trigger for Nuclio functions.

    Allows consuming messages from RabbitMQ queues or topic-based routing.

    See https://docs.nuclio.io/en/latest/reference/triggers/rabbitmq.html for more details.

    Example usage::

        trigger = RabbitMQTrigger(
            url="amqp://rabbitmq-host:5672",
            exchange_name="my-exchange",
            queue_name="my-queue",
            username="user",
            password="pass",
        )
        function.add_trigger("my-rabbitmq-trigger", trigger)

    Or with topics (routing keys)::

        trigger = RabbitMQTrigger(
            url="amqp://rabbitmq-host:5672",
            exchange_name="my-exchange",
            topics=["key1", "key2"],
        )

    Or using a datastore profile::

        trigger = RabbitMQTrigger(url="ds://my-rabbitmq-profile")

    When using a datastore profile (ds:// URL), all parameters from the profile
    are used as defaults. Any parameter explicitly passed will override the
    corresponding profile value, including falsy values like 0 or False::

        # Profile has prefetch_count=10, but explicit 0 overrides it
        trigger = RabbitMQTrigger(
            url="ds://my-rabbitmq-profile",
            prefetch_count=0,  # Overrides profile's prefetch_count=10
        )
    """

    kind = "rabbit-mq"

    def __init__(
        self,
        url: str,
        exchange_name: Optional[str] = None,
        queue_name: Optional[str] = None,
        topics: Optional[list[str]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        prefetch_count: Optional[int] = None,
        durable_exchange: Optional[bool] = None,
        durable_queue: Optional[bool] = None,
        on_error: Optional[str] = None,
        requeue_on_error: Optional[bool] = None,
        reconnect_duration: Optional[str] = None,
        reconnect_interval: Optional[str] = None,
        num_workers: Optional[int] = None,
        worker_termination_timeout: Optional[str] = None,
    ):
        """
        Initialize a RabbitMQ trigger.

        :param url:                       RabbitMQ connection URL in AMQP format
                                          (e.g., 'amqp://host:port' or 'amqp://user:pass@host:port')
                                          or a datastore profile URL (e.g., 'ds://profile-name')
        :param exchange_name:             The exchange that contains the queue
        :param queue_name:                Specific queue to consume from. Mutually exclusive
                                          with topics.
        :param topics:                    List of topics (routing keys) to subscribe to. Creates
                                          a unique queue and binds it to these routing keys.
                                          Mutually exclusive with queue_name.
        :param username:                  RabbitMQ username (can also be embedded in URL)
        :param password:                  RabbitMQ password (can also be embedded in URL)
        :param prefetch_count:            Broker channel prefetch limit (0 = unlimited)
        :param durable_exchange:          Whether the exchange should survive broker restart
        :param durable_queue:             Whether the queue should survive broker restart
        :param on_error:                  Error handling strategy: 'ack' or 'nack'
        :param requeue_on_error:          Whether to requeue failed messages (when on_error='nack')
        :param reconnect_duration:        Total time to attempt reconnection (e.g., '5m')
        :param reconnect_interval:        Time between reconnection attempts (e.g., '15s')
        :param num_workers:               Number of workers processing messages concurrently
        :param worker_termination_timeout: Timeout for worker termination (e.g., '10s')
        """
        # Handle datastore profile URL - merge profile values with explicit params
        if url.startswith("ds://"):
            profile = mlrun.datastore.datastore_profile.datastore_profile_read(url)
            if not isinstance(
                profile, mlrun.datastore.datastore_profile.DatastoreProfileRabbitMQ
            ):
                raise ValueError(
                    f"Unexpected datastore profile type: {profile.type}. "
                    "Only DatastoreProfileRabbitMQ is supported."
                )
            attrs = profile.attributes()
            url = attrs["url"]
            exchange_name = _first_not_none(exchange_name, attrs.get("exchange_name"))
            queue_name = _first_not_none(queue_name, attrs.get("queue_name"))
            topics = _first_not_none(topics, attrs.get("topics"))
            username = _first_not_none(username, attrs.get("username"))
            password = _first_not_none(password, attrs.get("password"))
            prefetch_count = _first_not_none(
                prefetch_count, attrs.get("prefetch_count")
            )
            durable_exchange = _first_not_none(
                durable_exchange, attrs.get("durable_exchange")
            )
            durable_queue = _first_not_none(durable_queue, attrs.get("durable_queue"))
            on_error = _first_not_none(on_error, attrs.get("on_error"))
            requeue_on_error = _first_not_none(
                requeue_on_error, attrs.get("requeue_on_error")
            )
            reconnect_duration = _first_not_none(
                reconnect_duration, attrs.get("reconnect_duration")
            )
            reconnect_interval = _first_not_none(
                reconnect_interval, attrs.get("reconnect_interval")
            )
            num_workers = _first_not_none(num_workers, attrs.get("num_workers"))
            worker_termination_timeout = _first_not_none(
                worker_termination_timeout, attrs.get("worker_termination_timeout")
            )

        # Extract credentials from URL if not provided explicitly
        creds = extract_credentials_from_url(url)
        url = creds.url
        username = _first_not_none(username, creds.username)
        password = _first_not_none(password, creds.password)

        # Validate
        if queue_name and topics:
            raise ValueError("Cannot specify both queue_name and topics. Choose one.")
        if on_error is not None and on_error not in ("ack", "nack"):
            raise ValueError(f"on_error must be 'ack' or 'nack', got '{on_error}'")

        # Build the trigger structure
        struct = {"kind": self.kind, "url": url, "attributes": {}}
        attrs = struct["attributes"]

        if username is not None:
            struct["username"] = username
        if password is not None:
            struct["password"] = password
        if num_workers is not None:
            struct["numWorkers"] = num_workers
        if worker_termination_timeout is not None:
            struct["workerTerminationTimeout"] = worker_termination_timeout
        if exchange_name is not None:
            attrs["exchangeName"] = exchange_name
        if queue_name is not None:
            attrs["queueName"] = queue_name
        if topics is not None:
            attrs["topics"] = topics
        if reconnect_duration is not None:
            attrs["reconnectDuration"] = reconnect_duration
        if reconnect_interval is not None:
            attrs["reconnectInterval"] = reconnect_interval
        if prefetch_count is not None:
            attrs["prefetchCount"] = prefetch_count
        if durable_exchange is not None:
            attrs["durableExchange"] = durable_exchange
        if durable_queue is not None:
            attrs["durableQueue"] = durable_queue
        if on_error is not None:
            attrs["onError"] = on_error
        if requeue_on_error is not None:
            attrs["requeueOnError"] = requeue_on_error

        super().__init__(struct)
