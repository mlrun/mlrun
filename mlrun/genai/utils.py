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
