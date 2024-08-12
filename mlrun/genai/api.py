# Copyright 2023 Iguazio
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

from typing import List, Union

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mlrun.genai.client import Client
from mlrun.genai.config import config
from mlrun.genai.data.doc_loader import get_data_loader, get_loader_obj
from mlrun.genai.schema import QueryItem, Document

app = FastAPI()

# Add CORS middleware, remove in production
origins = ["*"]  # React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a router with a prefix
router = APIRouter(prefix="/api")

client = Client(base_url=config.api_url)


class AuthInfo(BaseModel):
    username: str
    token: str
    roles: List[str] = []


# placeholder for extracting the Auth info from the request
async def get_auth_user(
    request: Request, x_username: Union[str, None] = Header(None)
) -> AuthInfo:
    """Get the user from the database"""
    token = request.cookies.get("Authorization", "")
    if x_username:
        return AuthInfo(username=x_username, token=token)
    else:
        return AuthInfo(username="guest@example.com", token=token)


@router.post("/data_sources/{data_source_name}/ingest")
async def ingest(
    data_source_name: str,
    database_kwargs: dict,
    loader: str,
    metadata: dict = None,
    document: Document = None,
    from_file: bool = False,
):
    """Ingest documents into the vector database"""
    data_loader = get_data_loader(
        config=config,
        client=client,
        data_source_name=data_source_name,
        database_kwargs=database_kwargs,
    )

    if from_file:
        with open(document.path, "r") as fp:
            lines = fp.readlines()
        for line in lines:
            path = line.strip()
            if path and not path.startswith("#"):
                loader_obj = get_loader_obj(path, loader_type=loader)
                data_loader.load(loader_obj, metadata=metadata, version=document.version)

    else:
        loader_obj = get_loader_obj(document.path, loader_type=loader)
        data_loader.load(loader_obj, metadata=metadata, version=document.version)
    return {"status": "ok"}


@router.post("/pipeline/{name}/run")
async def run_pipeline(
    request: Request,
    name: str,
    item: QueryItem,
    auth=Depends(get_auth_user),
):
    """This is the query command"""
    app_server = request.app.extra.get("app_server")
    if not app_server:
        raise ValueError("app_server not found in app")
    event = {
        "username": auth.username,
        "session_id": item.session_id,
        "query": item.question,
    }
    resp = app_server.run_pipeline(name, event)
    print(f"resp: {resp}")
    return resp
