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
#

import storey


class CacheEntry:
    def __init__(self, sequence, event):
        self.arrived = 1
        self.sequence = sequence
        self.events = [event]
        self.is_late = False

    def add_event(self, context, event, merge_key):
        if self.is_late:
            context.logger.warning(f"event id {self} arrived late")
            return None

        if context.verbose:
            context.logger.info(f"event id {merge_key} part {self.arrived} arrived")
        self.arrived += 1
        self.events.append(event)


class Merge(storey.Flow):
    def __init__(
        self,
        full_event: bool = None,
        key_path: str = None,
        max_behind: int = None,
        expected_num_events: int = None,
        **kwargs,
    ):
        """Merge multiple events based on event id or provided key path

        Users can subclass and overwrite the `get_join_key()` and `merge_function()` with custom logic

        :param key_path:   path to the event join key e.g. 'event["doc_id"]', default is the unique event id
        :param max_behind: max queue size to hold unmerged events,
                           oldest events will be dropped is queued longer (default=64)
        :param expected_num_events:  manually set the expected number of events per key
                                     (keep blank to auto detect from the graph)
        :param full_event: this step accepts the full Event object (body + metadata), not just body
        :param kwargs:     reserved for system use
        """
        self.key_path = key_path
        self.max_behind = max_behind
        self.expected_num_events = expected_num_events

        # use event.id (require full event) by default
        if full_event is None and not key_path:
            full_event = True
        self._graph_step = kwargs.pop("graph_step", None)
        super().__init__(full_event=full_event, **kwargs)

        self._uplinks = None
        self._cache = {}
        self._sequence = 0
        self._get_join_key = None
        self._queue_len = max_behind or 64  # default queue is 64 entries
        self._keys_queue = []

    def post_init(self, mode="sync"):
        # auto detect number of uplinks or use user specified value
        self._uplinks = self.expected_num_events or (
            len(self._graph_step.after) if self._graph_step else 0
        )

        # function to extract the join key from the event
        key_path = self.key_path or "event.id"
        self._get_join_key = eval("lambda event: " + key_path, {}, {})
        self._keys_queue = [None] * self._queue_len

    def get_join_key(self, event):
        """function extract the join key from the event, can be overwritten by sub class"""
        return self._get_join_key(event)

    async def _do(self, event):
        if event is storey.dtypes._termination_obj:
            return await self._do_downstream(storey.dtypes._termination_obj)
        else:
            element = self._get_event_or_body(event)
            fn_result = self._merge_events(element)
            if fn_result:
                mapped_event = self._user_fn_output_to_event(event, fn_result)
                await self._do_downstream(mapped_event)

    def _merge_events(self, event):
        # skip if only one uplink
        if self._uplinks <= 1:
            return event

        merge_key = self.get_join_key(event)
        if merge_key in self._cache:
            # old events with that key already exist (cached)
            entry: CacheEntry = self._cache[merge_key]
            entry.add_event(self.context, event, merge_key)

            if entry.arrived == self._uplinks:
                # expected number of events arrived, can merge them
                if self.context.verbose:
                    self.context.logger.info(
                        f"event {merge_key}, all {entry.arrived} parts arrived"
                    )
                del self._cache[merge_key]
                self._keys_queue[entry.sequence % self._queue_len] = None
                return self.merge_function(event, entry.events)
        else:
            # first time the key arrives
            self._cache[merge_key] = CacheEntry(self._sequence, event)
            queue_index = self._sequence % self._queue_len
            old_key = self._keys_queue[queue_index]

            if self.context.verbose:
                self.context.logger.info(f"new event id {merge_key} arrived")

            if old_key is not None:
                # reached max queued entries, need to drop the old event
                message = (
                    f"missing parts for event key {old_key} after a long wait, dropping"
                )
                self.context.logger.warning(message)
                if self._full_event:
                    self.context.push_error(
                        self._cache[old_key].events[0], f"{message}", source=self.name
                    )
                self._cache[old_key].is_late = True

            self._keys_queue[queue_index] = merge_key
            self._sequence += 1
            return None

    def merge_function(self, last_event, events):
        """logic to merge all gathered events to a result event, can be overwritten by sub class

        :param last_event: latest event
        :param events: all gathered events for the same key (including latest)
        :return: event
        """
        if self._full_event:
            last_event.body = [event.body for event in events]
            return last_event
        else:
            return events
