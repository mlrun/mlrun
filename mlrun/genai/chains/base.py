import asyncio

import storey

from ..schema import PipelineEvent


class ChainRunner(storey.Flow):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_async = asyncio.iscoroutinefunction(self._run)

    def _run(self, event: PipelineEvent):
        raise NotImplementedError()

    def __call__(self, event: PipelineEvent):
        return self._run(event)

    def post_init(self, mode="sync"):
        pass

    async def _do(self, event):
        if event is storey.dtypes._termination_obj:
            return await self._do_downstream(storey.dtypes._termination_obj)
        else:
            print("step name: ", self.name)
            element = self._get_event_or_body(event)
            if self._is_async:
                resp = await self._run(element)
            else:
                resp = self._run(element)
            if resp:
                for key, val in resp.items():
                    element.results[key] = val
                if "answer" in resp:
                    element.query = resp["answer"]
                mapped_event = self._user_fn_output_to_event(event, element)
                await self._do_downstream(mapped_event)


class SessionLoader(storey.Flow):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _do(self, event):
        if event is storey.dtypes._termination_obj:
            return await self._do_downstream(storey.dtypes._termination_obj)
        else:
            element = self._get_event_or_body(event)
            if isinstance(element, dict):
                element = PipelineEvent(**element)

            self.context.session_store.read_state(element)
            mapped_event = self._user_fn_output_to_event(event, element)
            await self._do_downstream(mapped_event)


class HistorySaver(ChainRunner):

    def __init__(
        self,
        answer_key: str = None,
        question_key: str = None,
        save_sources: str = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.answer_key = answer_key
        self.question_key = question_key
        self.save_sources = save_sources

    async def _run(self, event: PipelineEvent):
        question = (
            event.results[self.question_key]
            if self.question_key
            else event.original_query
        )
        sources = None
        if self.save_sources and "sources" in event.results:
            sources = [src.metadata for src in event.results["sources"]]
            event.results["sources"] = sources
        event.conversation.add_message("Human", question)
        event.conversation.add_message(
            "AI", event.results[self.answer_key or "answer"], sources
        )

        self.context.session_store.save(event)
        return event.results
