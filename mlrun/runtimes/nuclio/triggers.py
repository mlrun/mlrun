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

from typing import Optional
from urllib.parse import urlparse

from nuclio.triggers import NuclioTrigger


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
        :param exchange_name:             The exchange that contains the queue (required unless
                                          using a datastore profile)
        :param queue_name:                Specific queue to consume from. Either queue_name or
                                          topics must be specified, but not both.
        :param topics:                    List of topics (routing keys) to subscribe to. Creates
                                          a unique queue and binds it to these routing keys. Either
                                          queue_name or topics must be specified, but not both.
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
        # Handle datastore profile URL
        if url.startswith("ds://"):
            from mlrun.datastore.datastore_profile import (
                DatastoreProfileRabbitMQ,
                datastore_profile_read,
            )

            datastore_profile = datastore_profile_read(url)
            if not isinstance(datastore_profile, DatastoreProfileRabbitMQ):
                raise ValueError(
                    f"Unexpected datastore profile type: {datastore_profile.type}. "
                    "Only DatastoreProfileRabbitMQ is supported."
                )

            # Get attributes from profile, explicit params override profile values
            # Use 'is None' checks to properly handle falsy values like 0 and False
            attrs = datastore_profile.attributes()
            url = attrs["url"]
            if exchange_name is None:
                exchange_name = attrs.get("exchange_name")
            if queue_name is None:
                queue_name = attrs.get("queue_name")
            if topics is None:
                topics = attrs.get("topics")
            if username is None:
                username = attrs.get("username")
            if password is None:
                password = attrs.get("password")
            if prefetch_count is None:
                prefetch_count = attrs.get("prefetch_count")
            if durable_exchange is None:
                durable_exchange = attrs.get("durable_exchange")
            if durable_queue is None:
                durable_queue = attrs.get("durable_queue")
            if on_error is None:
                on_error = attrs.get("on_error")
            if requeue_on_error is None:
                requeue_on_error = attrs.get("requeue_on_error")
            if reconnect_duration is None:
                reconnect_duration = attrs.get("reconnect_duration")
            if reconnect_interval is None:
                reconnect_interval = attrs.get("reconnect_interval")
            if num_workers is None:
                num_workers = attrs.get("num_workers")
            if worker_termination_timeout is None:
                worker_termination_timeout = attrs.get("worker_termination_timeout")

        # Apply defaults for parameters still None after profile merge
        if prefetch_count is None:
            prefetch_count = 0
        if durable_exchange is None:
            durable_exchange = False
        if durable_queue is None:
            durable_queue = False
        if on_error is None:
            on_error = "nack"
        if requeue_on_error is None:
            requeue_on_error = False
        if reconnect_duration is None:
            reconnect_duration = "5m"
        if reconnect_interval is None:
            reconnect_interval = "15s"
        if num_workers is None:
            num_workers = 1
        if worker_termination_timeout is None:
            worker_termination_timeout = "10s"

        # Validate exchange_name is provided
        if not exchange_name:
            raise ValueError("exchange_name is required")

        # Validate that exactly one of queue_name or topics is specified
        if queue_name and topics:
            raise ValueError("Cannot specify both queue_name and topics. Choose one.")
        if not queue_name and not topics:
            raise ValueError("Must specify either queue_name or topics.")

        # Validate on_error value
        if on_error not in ("ack", "nack"):
            raise ValueError(f"on_error must be 'ack' or 'nack', got '{on_error}'")

        # Extract credentials from URL if not provided explicitly
        parsed_url = urlparse(url)
        if parsed_url.username and not username:
            username = parsed_url.username
        if parsed_url.password and not password:
            password = parsed_url.password

        # Build clean URL without credentials if they were embedded
        if parsed_url.username or parsed_url.password:
            # Reconstruct URL without credentials
            clean_url = f"{parsed_url.scheme}://{parsed_url.hostname}"
            if parsed_url.port:
                clean_url += f":{parsed_url.port}"
            if parsed_url.path:
                clean_url += parsed_url.path
            url = clean_url

        # Build the trigger structure
        struct = {
            "kind": self.kind,
            "url": url,
            "numWorkers": num_workers,
            "workerTerminationTimeout": worker_termination_timeout,
            "attributes": {
                "exchangeName": exchange_name,
                "reconnectDuration": reconnect_duration,
                "reconnectInterval": reconnect_interval,
                "prefetchCount": prefetch_count,
                "durableExchange": durable_exchange,
                "durableQueue": durable_queue,
            },
        }

        # Add queue_name or topics
        if queue_name:
            struct["attributes"]["queueName"] = queue_name
        if topics:
            struct["attributes"]["topics"] = topics

        # Add error handling configuration
        if on_error:
            struct["attributes"]["onError"] = on_error
        if requeue_on_error:
            struct["attributes"]["requeueOnError"] = requeue_on_error

        # Add credentials if provided
        if username:
            struct["username"] = username
        if password:
            struct["password"] = password

        super().__init__(struct)
