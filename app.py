"""Streamlit UI for the AI File Organizer.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter

import streamlit as st
from openai import OpenAI

from core.config import Settings
from core.mcp_client import open_path
from core.pipeline import flatten, organize
from core.rag import RagStore, answer_question

st.set_page_config(page_title="AI File Organizer", page_icon="🗂️", layout="wide")

settings = Settings()


def _root_cause(exc: BaseException) -> BaseException:
    """Unwrap a (Base)ExceptionGroup to the first underlying exception."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_rag(persist_dir: str, embed_model: str, _client: OpenAI) -> RagStore:
    return RagStore(persist_dir, _client, embed_model)


# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    if settings.openai_api_key:
        st.success("OpenAI API key loaded from .env")
    else:
        st.error("No OPENAI_API_KEY found. Copy .env.example to .env and set it.")
    st.text(f"Chat model:  {settings.chat_model}")
    st.text(f"Embeddings:  {settings.embed_model}")
    st.caption(
        "Files are read and moved by a local MCP filesystem server. "
        "Categories and file contents are indexed in a local RAG store."
    )

st.title("🗂️ AI File Organizer")
st.write(
    "Point this at a folder, choose how many files the AI should classify, and "
    "it will read each file (text, documents, and images), sort it into a "
    "two-level category folder, and move it there."
)

if not settings.openai_api_key:
    st.stop()

client = get_openai_client(settings.openai_api_key)
rag = get_rag(settings.rag_dir, settings.embed_model, client)

# --- Flatten nested folders -------------------------------------------------
FLATTEN_ACTIONS = {"Move": "move", "Copy": "copy"}

with st.expander("📁 Flatten nested folders (gather files from sub-folders first)"):
    st.write(
        "If your folder contains sub-folders, pull every file (at any depth) into "
        "one flat folder so the organizer can classify them all. Point the "
        "organizer at the destination folder afterwards."
    )
    with st.form("flatten_form"):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            flatten_source = st.text_input(
                "Source folder (with nested sub-folders)",
                placeholder=r"C:\Users\You\Messy",
            )
        with fcol2:
            flatten_dest = st.text_input(
                "Destination folder (flat)",
                placeholder=r"C:\Users\You\Messy\_flat",
            )
        flatten_action_label = st.radio(
            "Action",
            options=list(FLATTEN_ACTIONS.keys()),
            index=0,
            horizontal=True,
            help="Move relocates the nested files; Copy keeps the originals in place.",
        )
        flatten_submitted = st.form_submit_button("📥 Flatten into folder")

    if flatten_submitted:
        if not flatten_source or not os.path.isdir(flatten_source):
            st.error("Please enter a valid source folder.")
        elif not flatten_dest:
            st.error("Please enter a destination folder.")
        else:
            flat_progress = st.progress(0.0, text="Starting...")

            def _flat_progress(done: int, total: int, name: str) -> None:
                fraction = done / total if total else 1.0
                flat_progress.progress(fraction, text=f"({done}/{total}) {name}")

            try:
                with st.spinner("Gathering files from nested folders..."):
                    flat_results = asyncio.run(
                        flatten(
                            source=flatten_source,
                            dest_dir=flatten_dest,
                            action=FLATTEN_ACTIONS[flatten_action_label],
                            progress=_flat_progress,
                        )
                    )
                flat_progress.empty()
                if flat_results:
                    verb = FLATTEN_ACTIONS[flatten_action_label].title() + "d"
                    st.success(
                        f"{verb} {len(flat_results)} file(s) into {flatten_dest}"
                    )
                    st.dataframe(
                        [
                            {"File": r["filename"], "From": r["from"], "To": r["to"]}
                            for r in flat_results
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.warning("No files were found in nested folders.")
            except BaseException as exc:  # noqa: BLE001 - surface any runtime error
                flat_progress.empty()
                st.exception(_root_cause(exc))

# --- Organize form ----------------------------------------------------------
# Label shown in the UI -> sort key understood by the pipeline.
SORT_OPTIONS = {
    "Newest first": "newest",
    "Oldest first": "oldest",
    "Largest first": "largest",
    "Smallest first": "smallest",
    "Name (A–Z)": "name_asc",
    "Name (Z–A)": "name_desc",
}

# Label shown in the UI -> action understood by the pipeline.
ACTION_OPTIONS = {
    "Copy (keep originals)": "copy",
    "Move (relocate originals)": "move",
    "Preview only (no changes)": "preview",
}

with st.form("organize_form"):
    folder = st.text_input(
        "Folder to organize",
        placeholder=r"C:\Users\You\Downloads",
    )
    action_label = st.radio(
        "What should happen to the files?",
        options=list(ACTION_OPTIONS.keys()),
        index=0,
        horizontal=True,
        help="Copy keeps your originals, Move relocates them, Preview only shows the plan.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        limit = st.number_input(
            "How many files should the AI classify?",
            min_value=1,
            max_value=5000,
            value=10,
            step=1,
        )
    with col2:
        sort_label = st.selectbox(
            "Start organizing from",
            options=list(SORT_OPTIONS.keys()),
            index=0,
            help="Which files are picked first when the limit is smaller than the folder.",
        )
    with col3:
        output_name = st.text_input("Output subfolder name", value="Organized")
    recursive = st.checkbox(
        "Include files inside sub-folders (recursive)",
        value=True,
        help="Walk every sub-folder at any depth. Source-code files are always skipped so projects aren't disturbed.",
    )
    all_files = st.checkbox(
        "Classify ALL files (ignore the number above)",
        value=False,
        help="Process every eligible file in the folder, not just the number chosen.",
    )
    concurrency = st.slider(
        "Parallel workers",
        min_value=1,
        max_value=16,
        value=5,
        help="How many files to classify at the same time. Higher is faster but uses more tokens per minute (watch rate limits).",
    )
    submitted = st.form_submit_button("🚀 Organize", type="primary")

if submitted:
    if not folder or not os.path.isdir(folder):
        st.error("Please enter a valid, existing folder path.")
    else:
        output_base = os.path.join(folder, output_name)
        action = ACTION_OPTIONS[action_label]
        spinner_text = {
            "copy": "Classifying and copying files...",
            "move": "Classifying and moving files...",
            "preview": "Classifying files (preview, no changes)...",
        }[action]
        progress_bar = st.progress(0.0, text="Starting...")

        def _progress(done: int, total: int, name: str) -> None:
            fraction = done / total if total else 1.0
            progress_bar.progress(fraction, text=f"({done}/{total}) {name}")

        try:
            with st.spinner(spinner_text):
                results = asyncio.run(
                    organize(
                        folder=folder,
                        limit=None if all_files else int(limit),
                        output_base=output_base,
                        settings=settings,
                        oa_client=client,
                        rag=rag,
                        sort_by=SORT_OPTIONS[sort_label],
                        action=action,
                        recursive=recursive,
                        concurrency=int(concurrency),
                        progress=_progress,
                    )
                )
            progress_bar.empty()
            st.session_state["results"] = results
            if not results:
                st.warning("No files found to organize in that folder.")
        except BaseException as exc:  # noqa: BLE001 - surface any runtime error in UI
            progress_bar.empty()
            st.exception(_root_cause(exc))

# --- Results ----------------------------------------------------------------
results = st.session_state.get("results")
if results:
    ok = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]
    action_used = results[0].get("action", "copy")
    verb = {"copy": "Copied", "move": "Moved", "preview": "Previewed"}.get(
        action_used, "Organized"
    )
    st.subheader(f"✅ {verb} {len(ok)} file(s)")
    if action_used == "preview":
        st.info("Preview only — no files were changed. The destinations below are the plan.")
    if failed:
        st.warning(f"⚠️ {len(failed)} file(s) could not be processed (see below).")

    if ok:
        counts = Counter(r["category"] for r in ok)
        st.bar_chart(counts)

        dest_header = "Planned destination" if action_used == "preview" else "Destination"
        st.dataframe(
            [
                {
                    "File": r["filename"],
                    "Type": r["kind"],
                    "Category": r["category"],
                    "Summary": r["summary"],
                    dest_header: r.get("destination"),
                }
                for r in ok
            ],
            use_container_width=True,
            hide_index=True,
        )

    if failed:
        with st.expander(f"⚠️ {len(failed)} failed file(s)"):
            st.dataframe(
                [{"File": r["filename"], "Error": r["error"]} for r in failed],
                use_container_width=True,
                hide_index=True,
            )

# --- RAG Q&A ----------------------------------------------------------------
st.divider()
st.subheader("🔎 Ask about your organized files (RAG)")

index_col, clear_col = st.columns([4, 1])
with index_col:
    st.caption(f"{rag.file_count()} file(s) currently indexed.")
with clear_col:
    if st.button("🗑️ Clear index"):
        rag.clear()
        st.success("Index cleared.")
        st.rerun()

question = st.text_input(
    "Question", placeholder="e.g. Which files are invoices? Where are the cat memes?"
)
if st.button("Ask") and question:
    with st.spinner("Searching your files..."):
        sources = rag.search(question, n=5)
        answer = answer_question(client, settings.chat_model, question, sources)
    st.session_state["qa"] = {"answer": answer, "sources": sources}

qa = st.session_state.get("qa")
if qa:
    st.markdown(qa["answer"])
    sources = qa["sources"]
    if sources:
        with st.expander("Sources", expanded=True):
            for i, src in enumerate(sources):
                meta = src.get("metadata", {})
                path = meta.get("path") or ""
                st.markdown(
                    f"**{meta.get('filename', 'unknown')}** "
                    f"— _{meta.get('category', 'n/a')}_"
                )
                st.caption((src.get("content") or "")[:300])
                if path and os.path.exists(path):
                    open_col, folder_col = st.columns(2)
                    with open_col:
                        if st.button("📂 Open file", key=f"open_file_{i}"):
                            res = asyncio.run(open_path(path))
                            if res.get("error"):
                                st.error(res["error"])
                            else:
                                st.toast(f"Opened {meta.get('filename', path)}")
                    with folder_col:
                        if st.button("📁 Open folder", key=f"open_folder_{i}"):
                            res = asyncio.run(open_path(os.path.dirname(path)))
                            if res.get("error"):
                                st.error(res["error"])
                            else:
                                st.toast("Opened folder")
                    st.caption(path)
                elif path:
                    st.caption(f"⚠️ Not found on disk: {path}")
