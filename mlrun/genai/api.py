from typing import List, Union

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mlrun.genai.client import Client
from mlrun.genai.config import config
from mlrun.genai.data.doc_loader import get_data_loader, get_loader_obj
from mlrun.genai.schema import QueryItem

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
        return AuthInfo(username="yhaviv@gmail.com", token=token)


@router.post("/collections/{collection}/{path}/{loader}/ingest")
async def ingest(collection, path, loader, metadata, version, from_file):
    """Ingest documents into the vector database"""
    data_loader = get_data_loader(config, client=client, collection_name=collection)
    if from_file:
        with open(path, "r") as fp:
            lines = fp.readlines()
        for line in lines:
            path = line.strip()
            if path and not path.startswith("#"):
                loader_obj = get_loader_obj(path, loader_type=loader)
                data_loader.load(loader_obj, metadata=metadata, version=version)

    else:
        loader_obj = get_loader_obj(path, loader_type=loader)
        data_loader.load(loader_obj, metadata=metadata, version=version)


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


@router.get("/users")
async def list_users(user, email):
    """List users"""
    data = client.list_users(email, user, output_mode="short")
    return data


@router.get("/collections")
async def list_collections(owner, metadata):
    """List document collections"""
    data = client.list_collections(owner, metadata, output_mode="short").with_raise()
    return data


@router.put("/collection/{name}")
async def update_collection(name, owner, description, category, labels):
    """Create or update a document collection"""
    labels = fill_params(labels)

    # check if the collection exists, if it does, update it, otherwise create it
    collection_exists = client.get_collection(name)
    if collection_exists:
        client.update_collection(
            name=name,
            description=description,
            category=category,
            labels=labels,
        )
    else:
        client.create_collection(
            name=name,
            description=description,
            owner_name=owner,
            category=category,
            labels=labels,
        )


@router.get("/session/{session_id}")
async def get_session(session_id):
    """Get a chat session"""
    data = client.get_session(session_id)
    return data


@router.get("/sessions/")
def list_sessions(username, mode, last=None, created: str = None):
    """List chat sessions"""
    data = client.list_sessions(
        username=username,
        last=last,
        output_mode=mode,
        created_after=created,
    )
    return data


def fill_params(params, params_dict=None):
    params_dict = params_dict or {}
    for param in params:
        i = param.find("=")
        if i == -1:
            continue
        key, value = param[:i].strip(), param[i + 1 :].strip()
        if key is None:
            raise ValueError(f"cannot find param key in line ({param})")
        params_dict[key] = value
    if not params_dict:
        return None
    return params_dict
