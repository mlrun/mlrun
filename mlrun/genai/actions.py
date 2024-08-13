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


def ingest(collection_name, item: IngestItem):
    """This is the data ingestion command"""
    logger.debug(
        f"Running Data Ingestion: collection_name={collection_name}, path={item.path}, loader={item.loader}"
    )
    data_loader = get_data_loader(
        config,
        data_source_name=collection_name,
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
