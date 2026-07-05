"""Orchestrates the organize workflow: list -> extract -> classify -> move -> index."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Callable, Optional

from openai import OpenAI

from .classifier import classify
from .config import Settings
from .mcp_client import call_tool, mcp_session
from .rag import RagStore

ProgressFn = Callable[[int, int, str], None]

# Maps a sort key to (entry -> sort value, reverse?).
_SORTERS: dict[str, tuple[Callable[[dict], object], bool]] = {
    "newest": (lambda e: e.get("mtime", 0.0), True),
    "oldest": (lambda e: e.get("mtime", 0.0), False),
    "largest": (lambda e: e.get("size", 0), True),
    "smallest": (lambda e: e.get("size", 0), False),
    "name_asc": (lambda e: e.get("name", "").lower(), False),
    "name_desc": (lambda e: e.get("name", "").lower(), True),
}

# Source-code / project file extensions. These are never organized so that
# moving files can't break an existing codebase.
CODE_EXT = {
    ".py", ".pyw", ".pyx", ".ipynb", ".js", ".mjs", ".cjs", ".jsx", ".ts",
    ".tsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".cs", ".go",
    ".rb", ".php", ".rs", ".swift", ".kt", ".kts", ".scala", ".sh", ".bash",
    ".zsh", ".bat", ".ps1", ".psm1", ".sql", ".r", ".pl", ".pm", ".lua",
    ".dart", ".vue", ".svelte", ".gradle", ".groovy", ".m", ".mm", ".ex",
    ".exs", ".clj", ".cljs", ".hs", ".erl", ".elm", ".f", ".f90", ".jl",
    ".asm", ".s", ".vb", ".coffee", ".tf",
}


async def organize(
    *,
    folder: str,
    limit: Optional[int] = None,
    output_base: str,
    settings: Settings,
    oa_client: OpenAI,
    rag: RagStore,
    sort_by: str = "newest",
    action: str = "copy",
    recursive: bool = True,
    concurrency: int = 5,
    progress: Optional[ProgressFn] = None,
) -> list[dict]:
    """Classify files in ``folder`` and place them under ``output_base``.

    Files are ordered by ``sort_by``; the first ``limit`` are processed. When
    ``limit`` is None or <= 0, every eligible file is processed.

    ``action`` controls what happens to each file:

    * ``"copy"``    — copy into the category folder, leave the original in place.
    * ``"move"``    — move into the category folder.
    * ``"preview"`` — classify only; nothing is written to disk.

    When ``recursive`` is True, files inside sub-folders are included too (the
    output folder itself is always skipped). Up to ``concurrency`` files are
    processed in parallel.

    Returns a list of per-file result dictionaries.
    """
    results: list[dict] = []
    output_base = os.path.abspath(output_base)
    completed = False

    try:
        async with mcp_session() as session:
            if recursive:
                entries = await call_tool(
                    session,
                    "list_files_recursive",
                    {"directory": folder, "skip_dir": output_base},
                )
            else:
                entries = await call_tool(
                    session, "list_files", {"directory": folder}
                )
            if isinstance(entries, dict) and entries.get("error"):
                raise RuntimeError(entries["error"])

            # Keep files only; never re-process anything inside the output
            # folder, and never touch source-code files (may be part of a project).
            output_prefix = output_base + os.sep
            files = [
                e
                for e in entries
                if not e["is_dir"]
                and e["ext"] not in CODE_EXT
                and os.path.abspath(e["path"]) != output_base
                and not os.path.abspath(e["path"]).startswith(output_prefix)
            ]

            # Order by the chosen strategy, then take the first ``limit`` files
            # (or all of them when no positive limit was given).
            key, reverse = _SORTERS.get(sort_by, _SORTERS["newest"])
            files.sort(key=key, reverse=reverse)
            if limit and limit > 0:
                files = files[:limit]

            used_categories: set[str] = set()
            total = len(files)
            results_arr: list[Optional[dict]] = [None] * total

            sem = asyncio.Semaphore(max(1, concurrency))
            rag_lock = asyncio.Lock()
            progress_lock = asyncio.Lock()
            progress_state = {"done": 0}

            async def _bump(name: str) -> None:
                if progress:
                    async with progress_lock:
                        progress_state["done"] += 1
                        progress(progress_state["done"], total, name)

            async def _process(index: int, entry: dict) -> None:
                async with sem:
                    try:
                        content = (
                            await call_tool(
                                session,
                                "extract_content",
                                {
                                    "path": entry["path"],
                                    "max_chars": settings.max_text_chars,
                                    "max_image_dim": settings.max_image_dim,
                                },
                            )
                            or {}
                        )
                        kind = content.get("kind", "unknown")
                        text = content.get("text")

                        # RAG reads/writes touch ChromaDB, which is not safe for
                        # concurrent access, so serialize them with rag_lock.
                        async with rag_lock:
                            suggestions = await asyncio.to_thread(
                                rag.suggest_categories, text or entry["name"]
                            )
                            known = sorted(used_categories) + [
                                c for c in suggestions if c not in used_categories
                            ]

                        # The slow part (gpt-5) runs in a thread so multiple
                        # files classify in parallel.
                        top, sub, summary = await asyncio.to_thread(
                            classify,
                            oa_client,
                            settings.chat_model,
                            filename=entry["name"],
                            ext=entry["ext"],
                            kind=kind,
                            text=text,
                            image_base64=content.get("image_base64"),
                            image_mime=content.get("mime"),
                            known_categories=known[:12],
                        )

                        category_path = f"{top}/{sub}" if sub else top
                        dest_dir = (
                            os.path.join(output_base, top, sub)
                            if sub
                            else os.path.join(output_base, top)
                        )

                        # Apply the chosen action.
                        if action == "move":
                            moved = await call_tool(
                                session,
                                "move_file",
                                {"source": entry["path"], "dest_dir": dest_dir},
                            ) or {}
                            if moved.get("error"):
                                raise RuntimeError(moved["error"])
                            new_path = moved.get("moved_to")
                        elif action == "copy":
                            copied = await call_tool(
                                session,
                                "copy_file",
                                {"source": entry["path"], "dest_dir": dest_dir},
                            ) or {}
                            if copied.get("error"):
                                raise RuntimeError(copied["error"])
                            new_path = copied.get("copied_to")
                        else:  # preview — do not touch the filesystem
                            new_path = os.path.join(dest_dir, entry["name"])

                        # Index into RAG (content for text, summary otherwise).
                        index_text = text or summary or entry["name"]
                        async with rag_lock:
                            await asyncio.to_thread(
                                rag.add_file,
                                str(uuid.uuid4()),
                                index_text,
                                {
                                    "filename": entry["name"],
                                    "category": category_path,
                                    "path": new_path or entry["path"],
                                    "summary": summary[:500],
                                },
                            )
                            await asyncio.to_thread(
                                rag.remember_category,
                                category_path,
                                summary or entry["name"],
                            )
                            used_categories.add(category_path)

                        results_arr[index] = {
                            "filename": entry["name"],
                            "kind": kind,
                            "category": category_path,
                            "summary": summary,
                            "action": action,
                            "destination": new_path,
                            "error": None,
                        }
                    except Exception as exc:  # noqa: BLE001 - skip bad files, keep going
                        results_arr[index] = {
                            "filename": entry["name"],
                            "kind": "error",
                            "category": None,
                            "summary": "",
                            "action": action,
                            "destination": None,
                            "error": str(exc),
                        }
                    await _bump(entry["name"])

            await asyncio.gather(*(_process(i, e) for i, e in enumerate(files)))
            results = [r for r in results_arr if r is not None]

            completed = True
    except BaseException:
        # A failure purely during MCP subprocess teardown (a known Windows
        # quirk) after every file was processed is non-fatal — keep results.
        if not completed:
            raise

    return results


async def flatten(
    *,
    source: str,
    dest_dir: str,
    action: str = "move",
    progress: Optional[ProgressFn] = None,
) -> list[dict]:
    """Gather every file nested under ``source`` (at any depth) into the single
    flat folder ``dest_dir``.

    ``action`` is ``"move"`` (default) or ``"copy"``. The destination folder is
    skipped so files already there are never re-processed. Name collisions get a
    numeric suffix. Returns a list of {filename, from, to} dictionaries.
    """
    results: list[dict] = []
    dest_dir = os.path.abspath(dest_dir)
    completed = False

    try:
        async with mcp_session() as session:
            entries = await call_tool(
                session,
                "list_files_recursive",
                {"directory": source, "skip_dir": dest_dir},
            )
            if isinstance(entries, dict) and entries.get("error"):
                raise RuntimeError(entries["error"])

            # Ignore any files already sitting directly in the destination folder.
            files = [
                e
                for e in entries
                if not e["is_dir"] and os.path.dirname(e["path"]) != dest_dir
            ]
            total = len(files)

            for index, entry in enumerate(files):
                tool = "copy_file" if action == "copy" else "move_file"
                try:
                    result = await call_tool(
                        session,
                        tool,
                        {"source": entry["path"], "dest_dir": dest_dir},
                    ) or {}
                    if result.get("error"):
                        raise RuntimeError(result["error"])
                    new_path = result.get("copied_to") or result.get("moved_to")
                    results.append(
                        {
                            "filename": entry["name"],
                            "from": entry["path"],
                            "to": new_path,
                            "error": None,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - skip bad files, keep going
                    results.append(
                        {
                            "filename": entry["name"],
                            "from": entry["path"],
                            "to": None,
                            "error": str(exc),
                        }
                    )

                if progress:
                    progress(index + 1, total, entry["name"])

            completed = True
    except BaseException:
        # Tolerate MCP subprocess teardown noise once everything is processed.
        if not completed:
            raise

    return results

