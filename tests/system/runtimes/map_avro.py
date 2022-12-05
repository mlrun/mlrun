# Copyright 2022 Iguazio
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
#

# mlrun: start-code

import io
import json

import avro
import avro.io
import avro.schema
from storey import MapClass


class MyMap(MapClass):

    schema = json.dumps(
        {
            "namespace": "example.avro",
            "type": "record",
            "name": "User",
            "fields": [
                {"name": "ticker", "type": "string"},
                {"name": "name", "type": ["string", "null"]},
                {"name": "price", "type": ["int", "null"]},
            ],
        }
    )
    AVRO_SCHEMA = avro.schema.parse(schema)


    def do(self, event):
        self.context.logger.info(
            f"MyMap-1: event = {event.body} event_type={type(event.body)}"
        )

        reader = avro.io.DatumReader(self.AVRO_SCHEMA)

        bytes_reader = io.BytesIO(event.body)
        decoder = avro.io.BinaryDecoder(bytes_reader)

        record = reader.read(decoder)
        self.context.logger.info(
            f"MyMap-2: record={record}, type(record)={type(record)}"
        )
        event.key = record["ticker"]
        event.body = record
        return event


# mlrun: end-code
