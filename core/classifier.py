"""File classification using an OpenAI vision-capable chat model."""

from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

from .oai_retry import with_retries

_SYSTEM_PROMPT = (
    "You are a meticulous file-organization assistant. You classify a single "
    "file into a clean folder hierarchy with AT MOST two levels: a broad "
    "top_category and an optional, more specific sub_category. Prefer reusing "
    "an existing category from the provided list when it fits, to keep the "
    "hierarchy consistent. Use short, human-readable Title Case names and do "
    "not put slashes inside a single category name. Only add a sub_category "
    "when it provides meaningful separation (for example Images/Memes vs "
    "Images/Animals, or Documents/Invoices vs Documents/Contracts). "
    "Respond ONLY with a JSON object of the form: "
    '{"top_category": string, "sub_category": string or null, "summary": string}. '
    "summary is one concise sentence describing the file's content."
)


def _sanitize(name: Optional[str]) -> str:
    """Turn an arbitrary label into a safe single-segment folder name."""
    if not name:
        return "Uncategorized"
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:60] or "Uncategorized"


def _build_user_text(filename: str, ext: str, known: list[str], payload: str) -> str:
    known_text = ", ".join(known) if known else "none yet"
    return (
        f"Filename: {filename}\n"
        f"Extension: {ext or '(none)'}\n"
        f"Existing categories (reuse when appropriate): {known_text}\n\n"
        f"{payload}"
    )


def classify(
    client: OpenAI,
    model: str,
    *,
    filename: str,
    ext: str,
    kind: str,
    text: Optional[str] = None,
    image_base64: Optional[str] = None,
    image_mime: Optional[str] = None,
    known_categories: Optional[list[str]] = None,
) -> tuple[str, Optional[str], str]:
    """Classify one file. Returns (top_category, sub_category|None, summary)."""
    known = known_categories or []

    if kind == "image" and image_base64:
        prompt = _build_user_text(
            filename,
            ext,
            known,
            "This file is an image. Look at it carefully and classify it, using "
            "a nested sub_category to separate image kinds (e.g. Images/Memes, "
            "Images/Animals, Images/Screenshots, Images/People).",
        )
        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{image_base64}"},
            },
        ]
    else:
        snippet = (text or "").strip()
        payload = (
            f"File content (may be truncated):\n{snippet}"
            if snippet
            else "No readable content was extracted; classify from the filename."
        )
        user_content = _build_user_text(filename, ext, known, payload)

    # Reasoning-model families (gpt-5, o1/o3/o4) only accept the default
    # temperature, so only pin temperature=0 for models that support it.
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }
    # Reasoning-model families (gpt-5, o1/o3/o4) do not accept a custom
    # temperature. For this simple classification task, request minimal
    # reasoning so gpt-5 responds fast instead of "thinking" for seconds.
    lowered = model.lower()
    if lowered.startswith("gpt-5"):
        kwargs["reasoning_effort"] = "minimal"
    elif lowered.startswith(("o1", "o3", "o4")):
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["temperature"] = 0

    response = with_retries(lambda: client.chat.completions.create(**kwargs))

    data = json.loads(response.choices[0].message.content or "{}")
    top_raw = data.get("top_category") or "Uncategorized"
    sub_raw = data.get("sub_category")

    # If the model collapsed the hierarchy into "Top/Sub", split it back out.
    if "/" in top_raw and not sub_raw:
        parts = [p for p in top_raw.split("/") if p.strip()]
        top_raw = parts[0]
        sub_raw = parts[1] if len(parts) > 1 else None

    top = _sanitize(top_raw)
    sub = _sanitize(sub_raw) if sub_raw else None
    summary = (data.get("summary") or "").strip()
    return top, sub, summary
