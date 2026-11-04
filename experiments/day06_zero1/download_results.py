"""Download and verify one Day 06 export on the local computer (stdlib only, Python >= 3.11).

python experiments/day06_zero1/download_results.py --run-id a100_YYYYMMDDTHHMMSSZ
python experiments/day06_zero1/download_results.py --run-id pretest1 --verify-only outputs/day06_downloads/pretest1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_archive(archive: Path) -> None:
    fields = archive.with_name(archive.name + ".sha256").read_text(encoding="utf-8").strip().split()
    if (len(fields) != 2 or fields[1].lstrip("*") != archive.name
            or not re.fullmatch(r"[a-fA-F0-9]{64}", fields[0])):
        raise ValueError("Invalid archive checksum file")
    if sha256(archive) != fields[0].lower():
        raise ValueError("Archive SHA256 mismatch; download again")
    print("ARCHIVE CHECKSUM PASSED", flush=True)


def downloads_folder() -> Path:
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
                value, _ = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")
                return Path(os.path.expandvars(value))
        except OSError:
            pass
    return Path.home() / "Downloads"


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (not relative.parts or relative.is_absolute() or ".." in relative.parts
            or "\\" in name or ":" in name):
        raise ValueError(f"Unsafe relative path: {name!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes result directory: {name!r}")
    return path


def unpack(archive: Path, root: Path) -> None:
    # A new directory prevents merging two runs or overwriting existing results.
    root.mkdir(exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            target = safe_path(root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                with stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
            else:
                raise ValueError(f"Unexpected archive member type: {member.name}")


def verify(root: Path, rid: str, allow_incomplete: bool = False) -> None:
    session = root / f"session_{rid}"
    manifest = session / "SHA256SUMS"
    checked = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
        if not match:
            raise ValueError(f"Invalid manifest line: {line!r}")
        expected, relative = match.groups()
        if relative in checked:
            raise ValueError(f"Duplicate manifest entry: {relative}")
        path = safe_path(root, relative)
        if not path.is_file() or sha256(path) != expected.lower():
            raise ValueError(f"File missing or SHA256 mismatch: {relative}")
        checked.add(relative)
    if not checked:
        raise ValueError("Empty file manifest")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    exempt = {f"session_{rid}/SHA256SUMS", f"session_{rid}/archive_members.txt"}
    # Older local downloads kept the tarball beside the extracted run directories.
    archive = root / f"day06_{rid}.tar.gz"
    if archive.exists() or archive.with_name(archive.name + ".sha256").exists():
        verify_archive(archive)
        exempt.update({archive.name, archive.name + ".sha256"})
    if actual - exempt != checked:
        raise ValueError("Files on disk do not match the manifest")
    print(f"FILE CHECKSUMS PASSED: {len(checked)} files", flush=True)

    status = dict(line.split("=", 1) for line in (session / "status.txt").read_text().splitlines() if "=" in line)
    if status.get("exit_code") != "0" or not (session / "SUCCESS.txt").is_file():
        if allow_incomplete:
            print("INCOMPLETE RUN: available files restored and verified; inspect the session logs.")
            return
        raise ValueError("Run did not finish successfully; files are saved, inspect session logs. "
                         "Use --allow-incomplete to accept a verified partial backup.")
    parity = (session / "parity.log").read_text(encoding="utf-8")
    if "PARITY: ALL CHECKS PASSED" not in parity or parity.count("=> DETECTED by check") != 4:
        raise ValueError("Parity or negative controls did not pass")
    for stage, expected_rows in [("count", 6), ("timing", 9), ("slowlink", 4)]:
        rows = json.loads((root / f"{stage}_summary_{rid}.json").read_text())
        if len(rows) != expected_rows:
            raise ValueError(f"Wrong number of {stage} summary rows: {len(rows)}")
        if stage == "count" and not all(
            sum(row[k + "_per_step"] for k in ("AllReduce", "ReduceScatter", "AllGather")) > 0 for row in rows
        ):
            raise ValueError("No NCCL kernels detected in a count run")
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith("_" + rid) and p != session]
    if len(runs) != 52:
        raise ValueError(f"Expected 52 experiment directories, found {len(runs)}")
    for run in runs:
        json.loads((run / "metrics.json").read_text())
        if not (run / "result.pt").is_file():
            raise ValueError(f"Missing result.pt: {run}")
    traces = [p for run in runs for p in run.glob("trace_rank*.json")]
    if len(traces) != 12:
        raise ValueError(f"Expected 12 traces, found {len(traces)}")
    print("LOCAL VERIFICATION PASSED: 52 runs, 12 traces, 3 summary tables, 4 detected controls")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host", default="ssh4.vast.ai")
    parser.add_argument("--port", type=int, default=17591)
    parser.add_argument("--downloads", type=Path, default=None)
    parser.add_argument("--verify-only", type=Path, help="verify an already extracted results directory; no network")
    parser.add_argument("--skip-download", action="store_true", help="unpack and verify an already downloaded archive")
    parser.add_argument("--allow-incomplete", action="store_true", help="accept a checksummed backup of a failed run")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        parser.error("run-id must contain only letters, digits, underscores and hyphens")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]*", args.host) or not 1 <= args.port <= 65535:
        parser.error("invalid SSH host or port")
    if args.verify_only:
        verify(args.verify_only.resolve(), args.run_id, args.allow_incomplete)
        return
    destination = (args.downloads or downloads_folder()) / f"day06_{args.run_id}"
    destination.mkdir(parents=True, exist_ok=True)
    basename = f"day06_{args.run_id}.tar.gz"
    archive = destination / basename
    checksum = destination / (basename + ".sha256")
    print(f"Local backup: {destination}", flush=True)
    if not args.skip_download:
        for path in (archive, checksum):
            remote = f"root@{args.host}:/workspace/AI-Infra-Learning/outputs/day06_exports/{path.name}"
            partial = path.with_name(path.name + ".part")
            subprocess.run(["scp", "-o", "ConnectTimeout=15", "-P", str(args.port), remote, str(partial)], check=True)
            partial.replace(path)
    verify_archive(archive)
    root = destination / "results"
    if root.exists():
        raise FileExistsError(f"Extraction directory exists: {root}. Verify it with --verify-only; "
                              "use another --downloads directory for a fresh extraction.")
    unpack(archive, root)
    verify(root, args.run_id, args.allow_incomplete)
    print(f"Saved archive, checksum and extracted results: {destination}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
