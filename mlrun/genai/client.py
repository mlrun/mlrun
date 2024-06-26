import requests

from .config import logger


class Client:
    def __init__(self, base_url, username=None, token=None):
        self.base_url = base_url
        self.username = username or "guest"
        self.token = token

    def post_request(self, path, data=None, params=None, method="GET", files=None):
        # Construct the URL
        url = f"{self.base_url}/api/{path}"
        logger.debug(
            f"Sending {method} request to {url}, params: {params}, data: {data}"
        )

        # Make the request
        response = requests.request(
            method,
            url,
            json=data,
            params=(
                {k: v for k, v in params.items() if v is not None} if params else None
            ),
            headers={"x_username": self.username},
            files=files,
        )

        # Check the response
        if response.status_code == 200:
            # If the request was successful, return the JSON response
            return response.json()
        else:
            # If the request failed, raise an exception
            response.raise_for_status()

    def list_collections(self, owner=None, labels=None, output_mode=None):
        response = self.post_request(
            "collections",
            params={
                "owner": owner,
                "labels": labels,
                "mode": output_mode,
            },
        )
        return response["data"]

    def create_collection(self, name, **kwargs):
        response = self.post_request(f"collection/{name}", data=kwargs, method="POST")
        return response["success"]

    def run_pipeline(self, name, query, collection, session_id=None, filter=None):
        response = self.post_request(
            f"pipeline/{name or 'default'}/run",
            data={
                "question": query,
                "collection": collection,
                "session_id": session_id,
                "filter": filter,
            },
            method="POST",
        )
        data = response["data"]
        print(f"response: {response}")
        return data["answer"], data["sources"], data["returned_state"]

    # method to ingest a document
    def ingest(self, collection, path, loader, metadata=None, version=None):
        return self.post_request(
            f"collection/{collection}/ingest",
            data={
                "path": path,
                "loader": loader,
            },
            method="POST",
        )

    def get_session(self, session_id):
        response = self.post_request(f"session/{session_id}")
        return response["data"]

    def list_sessions(
        self, username=None, created_after=None, last=None, output_mode=None
    ):
        response = self.post_request(
            "sessions",
            params={
                "username": username,
                "created_after": created_after,
                "last": last,
                "mode": output_mode,
            },
        )
        return response["data"]

    def transcribe(self, audio_file):
        with open(audio_file, "rb") as af:
            response = self.post_request(
                "transcribe",
                files={"file": af},
                method="POST",
            )
            return response["data"]
