"""Stage released AF3 Champloo/Smorodina structures for Phase 2 (model_0 only).

Phase 2 needs predicted complex structures so the predictor-agnostic
``StructuralInterfaceScorer`` can compute clash/interface/contact/exposure/
packing geometry on the same cognate-vs-shuffled pairs Phase 1 scored from the
released AF3 ipTM matrix.

The AF3 archive on Zenodo (record 18390239,
``ab_ag_champloo_af3_real_shuffled_complexes.zip``) is ~31.6 GB and holds
556,248 PDB members named ``system_{i}_{j}_{vhhPDB}_{antigenPDB}_model_{k}.pdb``
(50 seeds, ``model_0``..``model_49``, per VHH-antigen pair). Downloading the
whole archive is explicitly out of scope. Instead this script reads the zip's
central directory via HTTP range requests, then range-extracts only one chosen
model (default ``model_0``) per matrix pair -- about 464 MB for full coverage of
the 8223 AF3 matrix pairs that have a structure.

Chain convention in the released PDBs (verified against the metadata
``vhh_length`` / ``antigen_length``): chain ``A`` is the VHH (binder), chain
``B`` is the antigen (target), regardless of the original crystal chain IDs.

Each chosen structure is written to ``<output-root>/<example_id>/rank1.pdb``
(``example_id = {VHH_PDB}__{ANTIGEN_PDB}``, upper-cased to match the ipTM
matrix), which is exactly the per-example layout ``StructuralInterfaceScorer``
expects. A manifest CSV records exactly what was downloaded.

Use::

    uv run python scripts/stage_champloo_structures.py \\
        --matrix <champloo>/iptm_confidence_scores/iptm_confidence_scores/af3_matrix_clean.csv \\
        --output-root <abdisc-data>/champloo/af3_structures \\
        --manifest data/staged/champloo/champloo_af3_structures_manifest.csv \\
        --dir-cache data/staged/champloo/af3_zip_directory.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Zenodo rate-limits guests; back off on 429/503 rather than aborting the run.
_RETRY_STATUS = frozenset({429, 503})
# Local file headers in this archive use a 28-byte extra field; over-read a bit
# so header + payload come back in a single range request.
_LOCAL_HEADER_PAD = 256

AF3_ZIP_URL = (
    "https://zenodo.org/records/18390239/files/ab_ag_champloo_af3_real_shuffled_complexes.zip"
)

_MEMBER_RE = re.compile(r"^system_(\d+)_(\d+)_([0-9A-Za-z]+)_([0-9A-Za-z]+)_model_(\d+)\.pdb$")


def parse_member_name(name: str) -> tuple[int, int, str, str, int] | None:
    """Parse ``system_{i}_{j}_{vhhPDB}_{antigenPDB}_model_{k}.pdb``.

    Returns ``(system_i, system_j, vhh_pdb, antigen_pdb, model)`` or ``None`` if
    the name does not match the released convention.
    """
    m = _MEMBER_RE.match(name)
    if m is None:
        return None
    i, j, vhh_pdb, antigen_pdb, model = m.groups()
    return int(i), int(j), vhh_pdb, antigen_pdb, int(model)


def example_id_for(vhh_pdb: str, antigen_pdb: str) -> str:
    """Pair example id, upper-cased to match the ipTM matrix PDB ids."""
    return f"{vhh_pdb.upper()}__{antigen_pdb.upper()}"


def index_model_entries(
    entries: list[dict[str, Any]], *, model: int
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index zip entries for one model by ``(vhh_pdb, antigen_pdb)`` (upper-cased).

    The first matching member wins when duplicate-PDB systems collapse to the
    same ipTM cell -- the released matrix cannot tell those systems apart, so a
    single representative structure is the faithful choice.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        parsed = parse_member_name(entry["name"])
        if parsed is None:
            continue
        _, _, vhh_pdb, antigen_pdb, mdl = parsed
        if mdl != model:
            continue
        key = (vhh_pdb.upper(), antigen_pdb.upper())
        if key not in out:
            out[key] = entry
    return out


def matrix_pairs(matrix_path: Path) -> list[tuple[str, str, int]]:
    """Enumerate finite ipTM cells as ``(vhh_pdb, antigen_pdb, label)``.

    ``label`` is 1 for the cognate diagonal (vhh_pdb == antigen_pdb) and 0 for
    off-diagonal shuffled non-cognate pairs. Empty / non-numeric / NaN cells are
    dropped, matching ``stage_champloo_pairs.py``.
    """
    with matrix_path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col_pdbs = [c.strip().upper() for c in header[1:]]
        out: list[tuple[str, str, int]] = []
        for raw in reader:
            row_pdb = raw[0].strip().upper()
            for col_pdb, value in zip(col_pdbs, raw[1:], strict=True):
                value = value.strip()
                if value == "":
                    continue
                try:
                    iptm = float(value)
                except ValueError:
                    continue
                if math.isnan(iptm):
                    continue
                out.append((row_pdb, col_pdb, 1 if row_pdb == col_pdb else 0))
    return out


def _urlopen_retry(req: urllib.request.Request, *, max_retries: int = 8) -> bytes:
    """Open a request, retrying on transient rate-limit / unavailable status."""
    delay = 1.0
    last_exc: Exception | None = None
    for _ in range(max_retries):
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in _RETRY_STATUS:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
            time.sleep(wait + random.uniform(0.0, 0.5))
            delay = min(delay * 2.0, 60.0)
        except urllib.error.URLError as exc:
            last_exc = exc
            time.sleep(delay + random.uniform(0.0, 0.5))
            delay = min(delay * 2.0, 60.0)
    raise RuntimeError(f"exhausted retries: {last_exc}")


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file backed by HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            self.size = int(resp.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = self.size - self.pos
        if size <= 0:
            return b""
        end = min(self.pos + size - 1, self.size - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = _urlopen_retry(req)
        self.pos += len(data)
        return data


def fetch_directory(url: str) -> dict[str, Any]:
    """Read the zip central directory via range requests (no payload download)."""
    range_file = HTTPRangeFile(url)
    zf = zipfile.ZipFile(range_file)
    entries = [
        {
            "name": zi.filename,
            "file_size": zi.file_size,
            "header_offset": zi.header_offset,
            "compress_size": zi.compress_size,
            "compress_type": zi.compress_type,
        }
        for zi in zf.infolist()
        if not zi.is_dir()
    ]
    return {"url": url, "size": range_file.size, "entries": entries}


def load_or_fetch_directory(url: str, cache_path: Path | None) -> dict[str, Any]:
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("url") == url:
            return cached
    directory = fetch_directory(url)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(directory))
    return directory


def _decompress(payload: bytes, compress_type: int) -> bytes:
    if compress_type == zipfile.ZIP_STORED:
        return payload
    return zlib.decompress(payload, -15)


def extract_member(url: str, entry: dict[str, Any]) -> bytes:
    """Range-fetch a single zip member and return its decompressed bytes.

    The local file header is 30 bytes + filename + extra field; its extra-field
    length can differ from the central-directory one, so it must be read from
    the local header. To avoid a second round-trip per file, over-read by
    ``_LOCAL_HEADER_PAD`` bytes so header + payload usually arrive in one range
    request; fall back to a precise second request if the extra field is larger
    than the pad.
    """
    header_offset = entry["header_offset"]
    compress_size = entry["compress_size"]
    over_read = 30 + _LOCAL_HEADER_PAD + compress_size
    end = header_offset + over_read - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={header_offset}-{end}"})
    blob = _urlopen_retry(req)
    fn_len, extra_len = struct.unpack("<HH", blob[26:30])
    payload_start = 30 + fn_len + extra_len
    payload_end = payload_start + compress_size
    if len(blob) >= payload_end:
        return _decompress(blob[payload_start:payload_end], entry["compress_type"])
    # Extra field larger than the pad: do a precise payload fetch.
    data_start = header_offset + payload_start
    data_end = data_start + compress_size - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={data_start}-{data_end}"})
    payload = _urlopen_retry(req)
    return _decompress(payload, entry["compress_type"])


def _stage_one(url: str, entry: dict[str, Any], dest: Path) -> None:
    raw = extract_member(url, entry)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)


_MANIFEST_FIELDS = (
    "example_id",
    "vhh_pdb",
    "antigen_pdb",
    "label",
    "member_name",
    "system_i",
    "system_j",
    "compress_size",
    "file_size",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--zip-url", type=str, default=AF3_ZIP_URL)
    parser.add_argument("--predictor", type=str, default="af3")
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--dir-cache", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, default=0, help="Stage only the first N pairs (0 = all)."
    )
    args = parser.parse_args()

    print(f"Reading central directory for {args.zip_url} ...", file=sys.stderr)
    directory = load_or_fetch_directory(args.zip_url, args.dir_cache)
    entries = directory["entries"]
    url = directory["url"]
    print(
        f"  {len(entries)} zip members; archive {directory['size'] / 1e9:.1f} GB",
        file=sys.stderr,
    )

    index = index_model_entries(entries, model=args.model)
    print(
        f"  {len(index)} (vhh,antigen) pairs have a model_{args.model} structure",
        file=sys.stderr,
    )

    pairs = matrix_pairs(args.matrix)
    if args.limit:
        pairs = pairs[: args.limit]
    n_pos = sum(p[2] for p in pairs)
    print(
        f"  matrix finite pairs: {len(pairs)} ({n_pos} cognate, {len(pairs) - n_pos} shuffled)",
        file=sys.stderr,
    )

    # Plan the work: (manifest_row, entry, dest) for pairs that have a structure.
    plan: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    manifest_rows: list[dict[str, Any]] = []
    missing = 0
    for vhh_pdb, antigen_pdb, label in pairs:
        entry = index.get((vhh_pdb, antigen_pdb))
        if entry is None:
            missing += 1
            continue
        parsed = parse_member_name(entry["name"])
        assert parsed is not None
        system_i, system_j, _, _, _ = parsed
        example_id = example_id_for(vhh_pdb, antigen_pdb)
        row = {
            "example_id": example_id,
            "vhh_pdb": vhh_pdb,
            "antigen_pdb": antigen_pdb,
            "label": label,
            "member_name": entry["name"],
            "system_i": system_i,
            "system_j": system_j,
            "compress_size": entry["compress_size"],
            "file_size": entry["file_size"],
        }
        manifest_rows.append(row)
        dest = args.output_root / example_id / "rank1.pdb"
        if not dest.is_file():
            plan.append((row, entry, dest))

    print(
        f"  staging {len(plan)} structures ({len(manifest_rows) - len(plan)} already present, "
        f"{missing} matrix pairs without a structure)",
        file=sys.stderr,
    )

    done = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_stage_one, url, entry, dest): row["example_id"]
            for row, entry, dest in plan
        }
        for fut in as_completed(futures):
            example_id = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                errors.append(f"{example_id}: {type(exc).__name__}: {exc}")
            else:
                done += 1
                if done % 500 == 0:
                    print(f"    staged {done}/{len(plan)}", file=sys.stderr)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_mb = sum(r["compress_size"] for r in manifest_rows) / 1e6
    print(
        f"Staged {done} new structures (errors={len(errors)}); manifest has "
        f"{len(manifest_rows)} rows (~{total_mb:.0f} MB compressed) -> {args.manifest}",
        file=sys.stderr,
    )
    for err in errors[:20]:
        print(f"  ERROR {err}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
