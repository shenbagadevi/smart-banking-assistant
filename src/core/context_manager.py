from typing import List, Any, Dict


def limit_context(documents: List[Any], max_chars: int = 12000) -> str:
    """
    Limits retrieved context before sending to the LLM.

    Args:
        documents: Iterable of document-like objects with `page_content` attribute.
        max_chars: Maximum concatenated characters to include.

    Returns:
        Concatenated context string truncated to `max_chars`.
    """
    context = ""
    for doc in documents:
        text = getattr(doc, "page_content", str(doc))
        if len(context) + len(text) > max_chars:
            break
        if context:
            context += "\n\n"
        context += text
    return context
