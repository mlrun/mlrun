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
    SCHEMA = avro.schema.parse(schema)

    def do(self, event):
        self.context.logger.info(
            f"MyMap-1: event = {event.body} event_type={type(event.body)}"
        )

        reader = avro.io.DatumReader(MyMap.SCHEMA)

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
