"""Local MCP (Model Context Protocol) server exposing filesystem tools.

This server runs as a subprocess and is spoken to over stdio by the
Streamlit app (see ``core/mcp_client.py``). It centralizes all filesystem
access and file-content extraction so the rest of the app never touches the
disk directly.

Run standalone for debugging:
    python mcp_server.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("filesystem-organizer")

# --- File type groups -------------------------------------------------------
TEXT_EXT = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json", ".yaml",
    ".yml", ".xml", ".html", ".htm", ".ini", ".cfg", ".toml", ".py", ".js",
    ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go",
    ".rb", ".php", ".sh", ".bat", ".ps1", ".sql", ".r", ".swift", ".kt",
    ".rs", ".scala", ".pl", ".lua",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}

# Binary / media files: never read as text (classify by filename instead).
BINARY_EXT = {
    # video
    ".mkv", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".ts", ".m2ts", ".3gp", ".ogv",
    # audio
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus",
    # archives / disk images
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".iso", ".dmg",
    # executables / binaries
    ".exe", ".msi", ".dll", ".bin", ".apk", ".deb", ".rpm", ".so",
    # heavy design / 3d binaries
    ".psd", ".ai", ".sketch", ".blend", ".obj", ".fbx", ".stl",
}


# --- Content extraction helpers --------------------------------------------
def _read_text(path: str, max_chars: int) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read(max_chars)


def _read_pdf(path: str, max_chars: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts: list[str] = []
    total = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n".join(parts)[:max_chars]


def _read_docx(path: str, max_chars: int) -> str:
    import docx

    document = docx.Document(path)
    text = "\n".join(p.text for p in document.paragraphs)
    return text[:max_chars]


def _read_image(path: str, max_dim: int) -> dict:
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"kind": "image", "mime": "image/png", "image_base64": encoded}


def _long(path: str) -> str:
    """On Windows, return an extended-length path so long paths (>260 chars) work."""
    if os.name == "nt":
        abspath = os.path.abspath(path)
        if not abspath.startswith("\\\\?\\"):
            return "\\\\?\\" + abspath
        return abspath
    return path


def _unique_target(target: str) -> str:
    """Return a non-clobbering destination path by adding a numeric suffix."""
    if not os.path.exists(_long(target)):
        return target
    base, ext = os.path.splitext(target)
    counter = 1
    while os.path.exists(_long(f"{base} ({counter}){ext}")):
        counter += 1
    return f"{base} ({counter}){ext}"


def _sniff_type(path: str) -> str:
    """Best-effort human-readable file type from magic bytes (dependency-free).

    Detects the *real* type regardless of extension and flags common installers,
    so files can be classified on evidence rather than a possibly-wrong filename.
    """
    try:
        with open(_long(path), "rb") as fh:
            head = fh.read(262144)  # 256 KB — enough to find installer markers
    except OSError:
        return ""
    if not head:
        return "empty file"

    sig = head[:16]
    low = head.lower()

    if sig.startswith(b"%PDF"):
        return "PDF document"
    if sig.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG image"
    if sig[:3] == b"\xff\xd8\xff":
        return "JPEG image"
    if sig[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF image"
    if sig[:2] == b"BM":
        return "BMP image"
    if sig[:4] == b"Rar!":
        return "RAR archive"
    if sig[:6] == b"\x37\x7a\xbc\xaf\x27\x1c":
        return "7-Zip archive"
    if sig[:4] == b"PK\x03\x04":
        return "ZIP-based file (archive, or Office/APK/JAR)"
    if sig[:4] == b"\xd0\xcf\x11\xe0":
        if b"windows installer" in low:
            return "Windows Installer package (MSI)"
        return "Microsoft OLE compound file (legacy Office/MSI)"
    if sig[:2] == b"MZ":
        if b"nullsoft" in low:
            return "Windows installer (NSIS setup)"
        if b"inno setup" in low:
            return "Windows installer (Inno Setup)"
        if b"installshield" in low:
            return "Windows installer (InstallShield)"
        if b"wise installation" in low:
            return "Windows installer (Wise)"
        return "Windows executable (.exe) — application or installer"
    if sig[:4] == b"fLaC":
        return "FLAC audio"
    if sig[:3] == b"ID3" or sig[:2] == b"\xff\xfb":
        return "MP3 audio"
    if sig[:4] == b"OggS":
        return "Ogg media"
    if head[4:8] == b"ftyp":
        return "MP4/MOV media (video or audio)"
    if sig[:4] == b"\x1aE\xdf\xa3":
        return "Matroska / WebM video"
    return ""


# --- Tools ------------------------------------------------------------------
@mcp.tool()
def list_files(directory: str) -> str:
    """List the immediate entries of a directory (non-recursive).

    Returns a JSON array of objects:
    {name, path, ext, size, is_dir, mtime, ctime}.
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return json.dumps({"error": f"Not a directory: {directory}"})

    entries: list[dict] = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        try:
            info = os.stat(full)
            is_dir = os.path.isdir(full)
            size = 0 if is_dir else info.st_size
        except OSError:
            continue
        entries.append(
            {
                "name": name,
                "path": os.path.abspath(full),
                "ext": os.path.splitext(name)[1].lower(),
                "size": size,
                "is_dir": is_dir,
                "mtime": info.st_mtime,
                "ctime": info.st_ctime,
            }
        )
    return json.dumps(entries)


@mcp.tool()
def list_files_recursive(directory: str, skip_dir: str = "") -> str:
    """Recursively list every file under a directory (at any depth).

    Directories themselves are not returned. ``skip_dir`` (if given) and its
    contents are excluded so a destination folder can be ignored. Each entry
    has the same shape as ``list_files`` with ``is_dir`` always False.
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return json.dumps({"error": f"Not a directory: {directory}"})

    skip_abs = os.path.abspath(skip_dir) if skip_dir else None
    entries: list[dict] = []
    for root, dirs, names in os.walk(directory):
        if skip_abs:
            # Do not descend into the skip directory.
            dirs[:] = [
                d
                for d in dirs
                if os.path.abspath(os.path.join(root, d)) != skip_abs
            ]
        for name in sorted(names):
            full = os.path.join(root, name)
            try:
                info = os.stat(full)
            except OSError:
                continue
            entries.append(
                {
                    "name": name,
                    "path": os.path.abspath(full),
                    "ext": os.path.splitext(name)[1].lower(),
                    "size": info.st_size,
                    "is_dir": False,
                    "mtime": info.st_mtime,
                    "ctime": info.st_ctime,
                }
            )
    return json.dumps(entries)


@mcp.tool()
def extract_content(path: str, max_chars: int = 8000, max_image_dim: int = 1024) -> str:
    """Extract classifiable content from a file.

    Returns a JSON object. For text-like files: {kind: "text", text}.
    For images: {kind: "image", mime, image_base64}. Otherwise {kind: "unknown"}.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return json.dumps({"kind": "error", "error": f"Not a file: {path}"})

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in BINARY_EXT:
            # Media/binary file: don't read the bytes; give a detected-type hint
            # so classification uses evidence, not just the (maybe wrong) name.
            return json.dumps(
                {"kind": "unknown", "text": "", "detected": _sniff_type(path)}
            )
        if ext in IMAGE_EXT:
            return json.dumps(_read_image(path, max_image_dim))
        if ext in PDF_EXT:
            return json.dumps({"kind": "text", "text": _read_pdf(path, max_chars)})
        if ext in DOCX_EXT:
            return json.dumps({"kind": "text", "text": _read_docx(path, max_chars)})
        if ext in TEXT_EXT:
            return json.dumps({"kind": "text", "text": _read_text(path, max_chars)})
        # Unknown extension: attempt a best-effort text read.
        try:
            return json.dumps({"kind": "text", "text": _read_text(path, max_chars)})
        except (UnicodeError, OSError):
            return json.dumps(
                {"kind": "unknown", "text": "", "detected": _sniff_type(path)}
            )
    except Exception as exc:  # noqa: BLE001 - report any parse failure to caller
        return json.dumps({"kind": "error", "error": str(exc)})


@mcp.tool()
def create_directory(path: str) -> str:
    """Create a directory (and any missing parents)."""
    try:
        path = os.path.abspath(path)
        os.makedirs(_long(path), exist_ok=True)
        return json.dumps({"created": path})
    except Exception as exc:  # noqa: BLE001 - report failure to caller as JSON
        return json.dumps({"error": str(exc)})


@mcp.tool()
def move_file(source: str, dest_dir: str) -> str:
    """Move ``source`` into ``dest_dir``, creating the directory if needed.

    Never overwrites an existing file (a numeric suffix is added instead).
    Returns {moved_to} or {error}.
    """
    try:
        source = os.path.abspath(source)
        dest_dir = os.path.abspath(dest_dir)
        if not os.path.isfile(_long(source)):
            return json.dumps({"error": f"Source is not a file: {source}"})

        os.makedirs(_long(dest_dir), exist_ok=True)
        target = _unique_target(os.path.join(dest_dir, os.path.basename(source)))
        shutil.move(_long(source), _long(target))
        return json.dumps({"moved_to": os.path.abspath(target)})
    except Exception as exc:  # noqa: BLE001 - report failure to caller as JSON
        return json.dumps({"error": str(exc)})


@mcp.tool()
def copy_file(source: str, dest_dir: str) -> str:
    """Copy ``source`` into ``dest_dir``, creating the directory if needed.

    The original is left in place. Never overwrites an existing file (a numeric
    suffix is added instead). Returns {copied_to} or {error}.
    """
    try:
        source = os.path.abspath(source)
        dest_dir = os.path.abspath(dest_dir)
        if not os.path.isfile(_long(source)):
            return json.dumps({"error": f"Source is not a file: {source}"})

        os.makedirs(_long(dest_dir), exist_ok=True)
        target = _unique_target(os.path.join(dest_dir, os.path.basename(source)))
        shutil.copy2(_long(source), _long(target))
        return json.dumps({"copied_to": os.path.abspath(target)})
    except Exception as exc:  # noqa: BLE001 - report failure to caller as JSON
        return json.dumps({"error": str(exc)})


def _os_open(path: str) -> None:
    """Open a file or folder with the operating system's default handler."""
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606 - Windows
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


@mcp.tool()
def open_path(path: str) -> str:
    """Open an existing file or folder in the OS default app / file explorer.

    Returns {opened} on success or {error} if the path is missing or fails.
    """
    try:
        path = os.path.abspath(path)
        if not os.path.exists(_long(path)):
            return json.dumps({"error": f"Path does not exist: {path}"})
        _os_open(path)
        return json.dumps({"opened": path})
    except Exception as exc:  # noqa: BLE001 - report failure to caller as JSON
        return json.dumps({"error": str(exc)})


if __name__ == "__main__":
    mcp.run()
