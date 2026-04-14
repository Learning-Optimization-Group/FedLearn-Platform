"""
update_txt_dumps.py — Generate LLM-friendly component dumps of the FedLearn repo.

For each component directory (backend, framework, frontend, fedlearn-desktop, client-docker),
writes a single .txt file under docs/llm_dumps/ that an external LLM can consume as context.

Output structure (per component):

  1. HEADER        - git branch + commit + generated-at, component path, summary stats
  2. TREE          - filtered tree overview (only files that will be dumped)
  3. MANIFEST      - sorted list: path, size, lines
  4. CONFIG        - Dockerfiles, build files, properties, package.json, pyproject, etc.
  5. SOURCE        - application source files
  6. TESTS         - anything under test/ spec/ __tests__/ or *_test.*/*.test.*
  7. DOCS          - markdown under the component (marked "may be outdated")

Design notes:
 - Generated protobuf stubs, build artefacts, lockfiles, binary blobs, data dumps, and
   the output directory itself are excluded so the context the LLM sees is authored code.
 - Secret masking is conservative: only the obviously-secret keys are redacted, so the
   LLM still sees URLs, ports, and hostnames (legitimate config context).
 - Files above --max-file-bytes are listed in the manifest but their bodies are elided.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DEFAULT_DUMPS_DIR: str = os.path.join(PROJECT_ROOT, "docs", "llm_dumps")

# component root -> output filename
MAPPINGS: Dict[str, str] = {
    os.path.join(PROJECT_ROOT, "backend"): "fedlearn_backend.txt",
    os.path.join(PROJECT_ROOT, "client-docker"): "fedlearn_client_docker.txt",
    os.path.join(PROJECT_ROOT, "framework"): "fedlearn_framework.txt",
    os.path.join(PROJECT_ROOT, "frontend"): "fedlearn_frontend.txt",
    os.path.join(PROJECT_ROOT, "fedlearn-desktop"): "fedlearn_desktop.txt",
}

ALLOWED_EXTENSIONS: Set[str] = {
    ".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".sh", ".md",
    ".yaml", ".yml", ".toml", ".json", ".sql", ".proto", ".properties",
    ".gradle", ".xml", ".conf", ".cfg", ".ini", ".css", ".scss",
}

ALLOWED_FILENAMES: Set[str] = {
    "Dockerfile", "Dockerfile.dev", "Dockerfile.prod",
    "package.json", "pom.xml", "build.gradle", "settings.gradle",
    "application.properties", "application-production.properties",
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "CONTRIBUTING.md", "README.md", ".env.example",
    "tsconfig.json", "vite.config.ts", "vite.config.js",
    "Makefile", "entrypoint.sh",
}

# Directories never traversed.
SKIP_DIRS: Set[str] = {
    ".git", ".github", ".idea", ".vscode",
    "__pycache__", "venv", ".venv", "env", "node_modules",
    "target", "build", "dist", "out", ".next",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    "egg-info", "FedLearn.egg-info", "fedlearn.egg-info",
    ".gradle", "apache-maven-3.9.6", "gradle-8.7", ".mvn",
    "logs", "data", "datasets", "cifar-10-batches-py",
    "coverage", "htmlcov", ".coverage",
    # Skip our own output directory so re-runs don't re-ingest prior dumps.
    "llm_dumps",
}

# Files explicitly excluded even if the extension/name would match.
# These tend to be either secret-bearing or not authored code.
BLACKLIST_FILES: Set[str] = {
    ".env", ".env.local", ".env.production", ".env.development",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb",
    "poetry.lock", "Pipfile.lock",
    # Generated protobuf stubs — keep the .proto source instead.
    "fedlearn_pb2.py", "fedlearn_pb2_grpc.py", "fedlearn_pb2.pyi",
    # Our own artefacts
    "requirements.txt.bak",
}

BLACKLIST_SUFFIXES: Tuple[str, ...] = (
    ".bak", ".orig", ".tmp", ".swp",
    ".pyc", ".pyo", ".class", ".jar", ".war",
    ".min.js", ".min.css",
    ".pkl", ".pth", ".pt", ".onnx", ".h5", ".safetensors",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg",
    ".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".xz",
    ".lock",
)

# Keys whose values are redacted in properties / env / yaml.
# Intentionally narrow so useful config (URLs, hostnames, ports) remains visible.
SENSITIVE_KEYWORDS: Tuple[str, ...] = (
    "secret", "password", "passwd", "pwd",
    "token", "credential", "private_key", "privatekey",
    "api_key", "apikey", "access_key", "accesskey",
    "client_secret", "clientsecret",
)

# Long base64/hex blobs likely to be keys. Masked inline inside .properties / .env.
_B64_PATTERN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{48,}={0,2}(?![A-Za-z0-9+/=])")
_AWS_ACCESS_KEY = re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")

# Section classification -------------------------------------------------------

CONFIG_FILENAMES: Set[str] = {
    "Dockerfile", "Dockerfile.dev", "Dockerfile.prod",
    "package.json", "pom.xml", "build.gradle", "settings.gradle",
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "tsconfig.json", "vite.config.ts", "vite.config.js",
    "Makefile", "entrypoint.sh", ".env.example",
}
CONFIG_EXTENSIONS: Set[str] = {
    ".properties", ".toml", ".yaml", ".yml", ".gradle", ".ini", ".cfg", ".conf",
}


def classify(rel_path: str) -> str:
    """Return one of: config, tests, docs, source."""
    parts = rel_path.replace(os.sep, "/").split("/")
    name = parts[-1]
    ext = os.path.splitext(name)[1].lower()

    # Tests
    test_markers = {"test", "tests", "spec", "specs", "__tests__"}
    if any(p in test_markers for p in parts[:-1]):
        return "tests"
    if name.endswith((".test.ts", ".test.tsx", ".test.js", ".test.jsx",
                      ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx")):
        return "tests"
    if name.startswith("test_") and ext == ".py":
        return "tests"

    # Docs
    if ext == ".md":
        return "docs"

    # Config
    if name in CONFIG_FILENAMES or ext in CONFIG_EXTENSIONS:
        return "config"

    return "source"


# Secret masking ---------------------------------------------------------------

def _redact_value_in_line(line: str) -> str:
    """Redact the value portion of a key=value or key: value line."""
    if "=" in line:
        prefix, _ = line.split("=", 1)
        return f"{prefix}=[REDACTED]"
    if ":" in line:
        prefix, _ = line.split(":", 1)
        return f"{prefix}: [REDACTED]"
    return line


def mask_secrets(content: str, filename: str) -> str:
    """Mask values in env / properties / yaml files when keys look sensitive.

    - key=value lines whose key matches a SENSITIVE_KEYWORD become key=[REDACTED]
    - base64-looking blobs (>=48 chars) are masked
    - AWS access key IDs are masked
    - Dynamic references (${FOO:default}) are preserved so structure is visible
    """
    ext = os.path.splitext(filename)[1].lower()
    is_env = filename.startswith(".env")
    structural = ext in (".properties", ".yaml", ".yml", ".env")
    if not structural and not is_env:
        # Still scrub AWS keys and long blobs in source files — cheap and paranoid.
        content = _AWS_ACCESS_KEY.sub("[REDACTED_AWS_KEY]", content)
        return content

    out_lines: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "//")):
            out_lines.append(line)
            continue

        match = re.match(r"^([^=: ]+)\s*[=:]\s*(.*)$", stripped)
        if match:
            key, value = match.groups()
            key_lower = key.lower()
            is_sensitive = any(kw in key_lower for kw in SENSITIVE_KEYWORDS)
            if is_sensitive and value.strip() and not value.strip().startswith("${"):
                line = _redact_value_in_line(line)
                out_lines.append(line)
                continue

        # Mask any stray long base64 blob even when key didn't match.
        line = _B64_PATTERN.sub("[REDACTED_BLOB]", line)
        line = _AWS_ACCESS_KEY.sub("[REDACTED_AWS_KEY]", line)
        out_lines.append(line)

    trailing = "\n" if content.endswith("\n") else ""
    return "\n".join(out_lines) + trailing


# File discovery ---------------------------------------------------------------

@dataclass
class FileEntry:
    abs_path: str
    rel_path: str        # relative to PROJECT_ROOT
    size_bytes: int
    line_count: int = 0
    section: str = "source"
    truncated: bool = False  # True if body was elided due to size cap


def _is_binary(path: str, probe_bytes: int = 2048) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(probe_bytes)
    except OSError:
        return True
    if b"\x00" in chunk:
        return True
    # Heuristic: if >30% of bytes are non-text, treat as binary.
    text_chars = bytes(range(32, 127)) + b"\n\r\t\b\f"
    non_text = sum(b not in text_chars for b in chunk)
    return len(chunk) > 0 and non_text / len(chunk) > 0.30


def collect_files(target_dir: str, max_file_bytes: int) -> List[FileEntry]:
    entries: List[FileEntry] = []
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for fname in sorted(files):
            if fname in BLACKLIST_FILES:
                continue
            if any(fname.endswith(sfx) for sfx in BLACKLIST_SUFFIXES):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ALLOWED_EXTENSIONS and fname not in ALLOWED_FILENAMES \
                    and not fname.startswith(".env"):
                continue

            abs_path = os.path.join(root, fname)
            try:
                size = os.path.getsize(abs_path)
            except OSError as e:
                logging.warning("Could not stat %s: %s", abs_path, e)
                continue

            if _is_binary(abs_path):
                continue

            rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
            entries.append(FileEntry(
                abs_path=abs_path,
                rel_path=rel_path,
                size_bytes=size,
                section=classify(rel_path),
                truncated=size > max_file_bytes,
            ))
    return entries


# Rendering --------------------------------------------------------------------

def _git(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git"] + cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def render_header(component_name: str, target_dir: str, entries: List[FileEntry]) -> str:
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    commit = _git(["rev-parse", "--short", "HEAD"]) or "unknown"
    dirty = _git(["status", "--porcelain"])
    dirty_marker = " (uncommitted changes present)" if dirty else ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections: Dict[str, int] = {}
    total_lines = 0
    total_bytes = 0
    for e in entries:
        sections[e.section] = sections.get(e.section, 0) + 1
        total_lines += e.line_count
        total_bytes += e.size_bytes

    lines = [
        "=" * 80,
        f"FedLearn Platform — LLM context dump: {component_name}",
        "=" * 80,
        f"Component path : {os.path.relpath(target_dir, PROJECT_ROOT)}",
        f"Generated at   : {now}",
        f"Git branch     : {branch}",
        f"Git commit     : {commit}{dirty_marker}",
        f"Files included : {len(entries)} "
        f"(config={sections.get('config', 0)}, source={sections.get('source', 0)}, "
        f"tests={sections.get('tests', 0)}, docs={sections.get('docs', 0)})",
        f"Total lines    : {total_lines:,}",
        f"Total size     : {total_bytes / 1024:.1f} KB",
        "",
        "Notes for the reader:",
        "  * Everything below is verbatim from this repo at the commit above.",
        "  * Generated protobuf stubs, build artefacts, lockfiles, and binaries are excluded.",
        "  * Markdown docs are included but may lag behind code — trust the source sections.",
        "  * Sensitive values in env/properties files are shown as [REDACTED].",
        "=" * 80,
        "",
    ]
    return "\n".join(lines)


def render_tree(target_dir: str, entries: List[FileEntry]) -> str:
    """Compact tree view of just the files we are dumping."""
    rel_base = os.path.relpath(target_dir, PROJECT_ROOT)
    tree: Dict[str, List[str]] = {}
    for e in entries:
        rel_from_component = os.path.relpath(e.abs_path, target_dir)
        parts = rel_from_component.split(os.sep)
        dirpath = os.sep.join(parts[:-1]) if len(parts) > 1 else "."
        tree.setdefault(dirpath, []).append(parts[-1])

    out = ["-- TREE " + "-" * 72, f"{rel_base}/"]
    for d in sorted(tree.keys()):
        indent = "  " if d == "." else "  " + "  " * d.count(os.sep)
        if d != ".":
            out.append(f"  {d}/")
        for fname in sorted(tree[d]):
            out.append(f"{indent}  {fname}")
    out.append("")
    return "\n".join(out)


def render_manifest(entries: List[FileEntry]) -> str:
    out = ["-- MANIFEST " + "-" * 68,
           f"{'path':<70}  {'bytes':>9}  {'lines':>6}  section"]
    for e in sorted(entries, key=lambda x: x.rel_path):
        out.append(f"{e.rel_path:<70}  {e.size_bytes:>9}  {e.line_count:>6}  {e.section}")
    out.append("")
    return "\n".join(out)


def _fence_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mapping = {
        "py": "python", "java": "java", "ts": "typescript", "tsx": "tsx",
        "js": "javascript", "jsx": "jsx", "sh": "bash", "md": "markdown",
        "yml": "yaml", "yaml": "yaml", "json": "json", "xml": "xml",
        "sql": "sql", "proto": "proto", "properties": "properties",
        "gradle": "groovy", "toml": "toml", "dockerfile": "dockerfile",
    }
    if not ext:
        # Dockerfile, Makefile, entrypoint.sh etc.
        if os.path.basename(path).lower().startswith("dockerfile"):
            return "dockerfile"
        if os.path.basename(path).lower() == "makefile":
            return "makefile"
        return "text"
    return mapping.get(ext, "text")


def render_file(entry: FileEntry, max_file_bytes: int) -> str:
    header = [
        "=" * 80,
        f"File: {entry.rel_path}",
        f"Size: {entry.size_bytes} bytes | Lines: {entry.line_count} | Section: {entry.section}",
        "=" * 80,
        "",
    ]
    if entry.truncated:
        header.append(
            f"[ elided: file exceeds --max-file-bytes ({max_file_bytes}). "
            "Open the file directly if detail is needed. ]"
        )
        header.append("")
        return "\n".join(header) + "\n"

    try:
        with open(entry.abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        return "\n".join(header) + f"[ error reading file: {e} ]\n\n"

    content = mask_secrets(content, os.path.basename(entry.abs_path))
    if content and not content.endswith("\n"):
        content += "\n"

    fence = _fence_for(entry.abs_path)
    body = [f"```{fence}", content.rstrip("\n"), "```", ""]
    return "\n".join(header) + "\n".join(body) + "\n"


# Section ordering. Docs emit last with an explicit staleness warning.
_SECTION_ORDER = ("config", "source", "tests", "docs")
_SECTION_TITLES = {
    "config": "CONFIG & BUILD",
    "source": "SOURCE",
    "tests": "TESTS",
    "docs": "DOCS (may lag behind code)",
}


def build_dump(target_dir: str, output_file: str, max_file_bytes: int) -> Tuple[int, int]:
    component_name = os.path.basename(target_dir.rstrip(os.sep)) or target_dir
    logging.info("Building dump for %s -> %s", component_name, output_file)

    entries = collect_files(target_dir, max_file_bytes)
    # Populate line counts lazily now, in bulk.
    for e in entries:
        if e.truncated:
            continue
        try:
            with open(e.abs_path, "r", encoding="utf-8", errors="replace") as f:
                e.line_count = sum(1 for _ in f)
        except OSError:
            e.line_count = 0

    header = render_header(component_name, target_dir, entries)
    tree = render_tree(target_dir, entries)
    manifest = render_manifest(entries)

    sections_text: List[str] = []
    for section in _SECTION_ORDER:
        section_entries = [e for e in entries if e.section == section]
        if not section_entries:
            continue
        sections_text.append(f"{'#' * 80}")
        sections_text.append(f"# {_SECTION_TITLES[section]}")
        sections_text.append(f"{'#' * 80}\n")
        for e in sorted(section_entries, key=lambda x: x.rel_path):
            sections_text.append(render_file(e, max_file_bytes))

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(header)
        out.write(tree)
        out.write("\n")
        out.write(manifest)
        out.write("\n")
        out.write("\n".join(sections_text))

    total_lines = sum(e.line_count for e in entries)
    return len(entries), total_lines


# Entry point ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM text dumps for each component.")
    parser.add_argument("--outdir", default=DEFAULT_DUMPS_DIR,
                        help="Directory where component dumps are written.")
    parser.add_argument("--target", default=None,
                        help="Optional single component path (relative to repo root).")
    parser.add_argument("--max-file-bytes", type=int, default=256 * 1024,
                        help="Files larger than this are listed but their bodies elided. "
                             "Default: 256 KB.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.target:
        target_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.target))
        if not os.path.isdir(target_path):
            logging.error("Target %s does not exist or is not a directory.", args.target)
            return
        slug = args.target.strip("/").replace("/", "_") or "root"
        mappings = {target_path: os.path.join(args.outdir, f"fedlearn_{slug}.txt")}
    else:
        mappings = {src: os.path.join(args.outdir, dst) for src, dst in MAPPINGS.items()}

    total_files = 0
    total_lines = 0
    for src, dst in mappings.items():
        if not os.path.isdir(src):
            logging.warning("Skipping missing directory: %s", src)
            continue
        files, lines = build_dump(src, dst, args.max_file_bytes)
        logging.info(" -> %s: %d files, %d lines", os.path.basename(dst), files, lines)
        total_files += files
        total_lines += lines

    logging.info("Done. %d files, %d lines across all dumps.", total_files, total_lines)


if __name__ == "__main__":
    main()
