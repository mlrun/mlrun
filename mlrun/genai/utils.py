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

def sources_to_text(sources):
    """Convert a list of sources to a string."""
    if not sources:
        return ""
    return "\nSource documents:\n" + "\n".join(
        f"- {get_title(source.metadata)} ({source.metadata['source']})"
        for source in sources
    )


def sources_to_md(sources):
    """Convert a list of sources to a Markdown string."""
    if not sources:
        return ""
    sources = {
        source.metadata["source"]: get_title(source.metadata) for source in sources
    }
    return "\n**Source documents:**\n" + "\n".join(
        f"- [{title}]({url})" for url, title in sources.items()
    )


def get_title(metadata):
    """Get title from metadata."""
    if "chunk" in metadata:
        return f"{metadata.get('title', '')}-{metadata['chunk']}"
    if "page" in metadata:
        return f"{metadata.get('title', '')} - page {metadata['page']}"
    return metadata.get("title", "")
