from llmapps.controller.model import ChatSession

from .schema import PipelineEvent


class SessionStore:
    def __init__(self, client):
        self.db_session = None
        self.client = client

    def get_db_session(self):
        if self.db_session is None:
            self.db_session = self.client.get_db_session()
        return self.db_session

    def read_state(self, event: PipelineEvent, db_session=None):
        close_session = True if db_session is None else False
        db_session = db_session or self.get_db_session()
        event.username = event.username or "guest"
        event.user = self.client.get_user(event.username, session=db_session)
        if event.session_id:
            chat_session = self.client.get_session(
                event.session_id, session=db_session
            ).data
            if chat_session:
                event.session = chat_session
                event.state = chat_session.state
                event.conversation = chat_session.to_conversation()
            else:
                self.client.create_session(
                    ChatSession(
                        name=event.session_id, username=event.username or "guest"
                    ),
                    session=db_session,
                )

        if close_session:
            db_session.close()

    def save(self, event: PipelineEvent, db_session=None):
        """Save the session and conversation to the database"""
        if event.session_id:
            close_session = True if db_session is None else False
            db_session = db_session or self.get_db_session()
            self.client.update_session(
                ChatSession(
                    name=event.session_id,
                    state=event.state,
                    history=event.conversation.to_list(),
                ),
                session=db_session,
            )

            if close_session:
                db_session.close()


def get_session_store(config):
    # todo: support different session stores
    if config.use_local_db:
        from llmapps.controller.sqlclient import client

        return SessionStore(client)

    raise NotImplementedError("Only local db is supported for now")
