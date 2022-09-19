# Copyright 2018 Iguazio
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
from datetime import datetime
from typing import List, Optional, Union

import pandas
import storey
from storey import Event
from storey.dtypes import _termination_obj


class SqlDBSourceStorey(storey.sources._IterableSource, storey.sources.WithUUID):
    """Use mongodb collection as input source for a flow.

    :parameter key_field: the primary key of the collection.
    :parameter time_field: column to be used as time for events.
    :parameter id_field: column to be used as ID for events.
    :parameter db_name: url string connection to sql database.
    :parameter collection_name: the name of the collection to access,
                                from the current database
    :parameter start_filter: If not None, the results will be filtered by partitions and 'filter_column' > start_filter.
                            Default is None (not in use)
    :parameter end_filter:  If not None, the results will be filtered by partitions 'filter_column' <= end_filter.
                            Default is None (not in use)

    """

    def __init__(
        self,
        db_path: str = None,
        collection_name: str = None,
        key_field: Optional[Union[str, List[str]]] = None,
        start_filter: Optional[datetime] = None,
        end_filter: Optional[datetime] = None,
        time_field: Optional[str] = None,
        id_field: Optional[str] = None,
        **kwargs,
    ):

        if key_field is not None:
            kwargs["key_field"] = key_field
        if time_field is not None:
            kwargs["time_field"] = time_field
        if id_field is not None:
            kwargs["id_field"] = id_field
        storey.sources._IterableSource.__init__(self, **kwargs)
        storey.sources.WithUUID.__init__(self)

        if not all([db_path, collection_name]):
            raise ValueError("cannot specify without db_path and collection_name args")

        self.collection_name = collection_name
        self.db_path = db_path

        self._key_field = key_field
        if isinstance(time_field, str) and "." in time_field:
            self._time_field = time_field.split(".")
        else:
            self._time_field = time_field
        if isinstance(id_field, str) and "." in id_field:
            self._id_field = id_field.split(".")
        else:
            self._id_field = id_field

    async def _run_loop(self):
        import sqlalchemy as db

        engine = db.create_engine(self.db_path)
        metadata = db.MetaData()
        connection = engine.connect()
        collection = db.Table(
            self.collection_name, metadata, autoload=True, autoload_with=engine
        )
        results = connection.execute(db.select([collection])).fetchall()
        df = pandas.DataFrame(results)
        df.columns = results[0].keys()
        df.set_index("index", inplace=True)
        connection.close()

        for namedtuple in df.itertuples():
            create_event = True
            body = namedtuple._asdict()
            body.pop("Index")
            key = None
            if self._key_field:
                if isinstance(self._key_field, list):
                    key = []
                    for key_field in self._key_field:
                        if key_field not in body or pandas.isna(body[key_field]):
                            create_event = False
                            break
                        key.append(body[key_field])
                else:
                    key = body[self._key_field]
                    if key is None:
                        create_event = False
            if create_event:
                time = None
                if self._time_field:
                    time = self.get_val_from_multi_dictionary(body, self._time_field)
                if self._id_field:
                    _id = self.get_val_from_multi_dictionary(body, self._id_field)
                else:
                    _id = self._get_uuid()
                event = Event(body, key=key, time=time, id=_id)
                await self._do_downstream(event)
            else:
                if self.context:
                    self.context.logger.error(
                        f"For {body} value of key {key_field} is None"
                    )
        return await self._do_downstream(_termination_obj)

    def get_val_from_multi_dictionary(self, event, field):
        for f in field:
            event = event[f]

        return event
