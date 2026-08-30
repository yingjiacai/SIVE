"""Small provenance manifest for long-running experiment outputs."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(source_root, *args):
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_manifest(output_dir, config, seeds, source_root="."):
    """Freeze resolved parameters and hashes of the experiment source files."""
    output_path = Path(output_dir)
    root = Path(source_root).resolve()
    source_files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json"}
        and not any(part in {"outputs", "data", ".git", "__pycache__"}
                    for part in path.parts)
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "seeds": list(seeds),
        "git_head": _git_value(root, "rev-parse", "HEAD"),
        "git_status_porcelain": _git_value(root, "status", "--porcelain"),
        "source_sha256": {
            str(path.relative_to(root)): _sha256(path) for path in source_files
        },
    }
    manifest_path = output_path / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
    return manifest_path
