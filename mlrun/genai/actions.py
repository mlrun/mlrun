from typing import List, Optional, Tuple

import openai
from pydantic import BaseModel

from mlrun.genai.config import config, logger
from mlrun.genai.data.doc_loader import get_data_loader, get_loader_obj
from mlrun.genai.schema import ApiResponse


class IngestItem(BaseModel):
    path: str
    loader: str
    metadata: Optional[List[Tuple[str, str]]] = None
    version: Optional[str] = None


def ingest(client, collection_name, item: IngestItem):
    """This is the data ingestion command"""
    logger.debug(
        f"Running Data Ingestion: collection_name={collection_name}, path={item.path}, loader={item.loader}"
    )
    data_loader = get_data_loader(
        config,
        client=client,
        collection_name=collection_name,
    )
    loader_obj = get_loader_obj(item.path, loader_type=item.loader)
    data_loader.load(loader_obj, metadata=item.metadata, version=item.version)
    return ApiResponse(success=True)


def transcribe_file(file_handler):
    """transcribe audio file using openai API"""
    logger.debug("Transcribing file")
    text = openai.Audio.transcribe("whisper-1", file_handler)
    print(text)
    return ApiResponse(success=True, data=text)
