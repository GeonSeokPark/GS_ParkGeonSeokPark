#!/usr/bin/env python3
"""Extract one JPEG per identity from a pinned VGGFace2-HQ mirror."""
import argparse
import concurrent.futures
import hashlib
import io
import re
import struct
import threading
import time
import zlib
from pathlib import Path

import requests
from PIL import Image

BASE = ("https://huggingface.co/datasets/RichardErkhov/VGGFace2-HQ/resolve/"
        "6771c0319f29a7b77e4f71ebfb09f7b40066e2cc/original/")
INDEX_CONTENT_RANGE = "bytes 9637293527-9809373270/9809373369"
INDEX_RANGE = "bytes=9637293527-9809373270"
INDEX_SIZE = 172079744
INDEX_SHA256 = "a05d0ba66133ec7cd1400cf4f4379c5c8fc6f4d9e4ae4074921285726aa51079"
CENTRAL = struct.Struct("<4s6H3I5H2I")
LOCAL = struct.Struct("<4s5H3I2H")
ENTRY = re.compile(rb"VGGface2_None_norm_512_true_bygfpgan/(n[0-9]+)/([A-Za-z0-9_.-]+\.jpg)")
SESSIONS = threading.local()


def resolve_archive(disk):
    suffix = f"z{disk + 1:02}" if disk < 4 else "zip"
    url = BASE + f"VGGface2_HQ.{suffix}"
    for attempt in range(3):
        with requests.get(url, headers={"Range": "bytes=0-511"}, timeout=20,
                          stream=True, allow_redirects=False) as response:
            if response.status_code == 429 and attempt < 2:
                delay = int(response.headers.get("Retry-After", "60"))
            else:
                response.raise_for_status()
                return response.headers.get("Location", url)
        print(f"Mirror rate limit. Retrying in {delay} seconds.", flush=True)
        for start in range(0, delay, 30):
            time.sleep(min(30, delay - start))
    raise RuntimeError("Unable to resolve archive.")


def read_index(path):
    cached = path.is_file()
    if cached:
        data = path.read_bytes()
    else:
        with requests.get(resolve_archive(4), headers={"Range": INDEX_RANGE},
                          timeout=60, stream=True) as response:
            if (response.status_code != 206
                    or response.headers.get("Content-Range") != INDEX_CONTENT_RANGE):
                raise ValueError("The mirror did not return the pinned ZIP index range.")
            data = response.raw.read(INDEX_SIZE)
    if hashlib.sha256(data).hexdigest() != INDEX_SHA256:
        raise ValueError(f"The ZIP index did not match the archive revision: {path}")
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def gallery(data, count):
    selected = {}
    pos = 0
    while pos < len(data):
        sig, _, _, flags, method, _, _, crc, compressed, raw, names, extras, comment, disk, _, _, offset = CENTRAL.unpack_from(data, pos)
        if sig != b"PK\x01\x02":
            raise ValueError(f"Bad central signature at {pos}")
        start = pos + CENTRAL.size
        name = data[start:start + names]
        pos = start + names + extras + comment
        match = ENTRY.fullmatch(name)
        if not match:
            continue
        identity, fname = match.groups()
        if identity in selected and selected[identity][0] <= fname:
            continue
        if offset == 0xFFFFFFFF:
            extra = data[start + names:start + names + extras]
            while extra:
                code, length = struct.unpack_from("<HH", extra)
                if code == 1:
                    if raw == 0xFFFFFFFF or compressed == 0xFFFFFFFF:
                        raise ValueError("Unexpected oversized image")
                    offset = struct.unpack_from("<Q", extra, 4)[0]
                    break
                extra = extra[4 + length:]
        selected[identity] = (fname, name, disk, offset, compressed, raw, crc, method, flags)
    print(f"Indexed {len(selected)} identity folders", flush=True)
    if len(selected) < count:
        raise ValueError("Not enough identities")
    return [(key, selected[key]) for key in sorted(selected)[:count]]


def valid_file(target, raw_size, expected_crc):
    if not target.is_file() or target.stat().st_size != raw_size:
        return False
    return zlib.crc32(target.read_bytes()) == expected_crc


def fetch(target, entry, archive_url):
    _, archived_name, _, offset, compressed, raw, crc, method, flags = entry
    if flags & 1 or method != 8:
        raise ValueError("Unexpected encrypted or non-deflate archive member")
    if not hasattr(SESSIONS, "session"):
        SESSIONS.session = requests.Session()
    end = offset + LOCAL.size + len(archived_name) + compressed + 511
    for attempt in range(4):
        try:
            headers = {"Range": f"bytes={offset}-{end}"}
            with SESSIONS.session.get(archive_url, headers=headers, timeout=20, stream=True) as response:
                if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise OSError(f"Expected range {offset}, received {response.status_code}")
                data = b"".join(response.iter_content(chunk_size=2**16))
            sig, _, _, _, _, _, _, _, _, nlen, extra = LOCAL.unpack_from(data)
            if sig != b"PK\x03\x04" or data[LOCAL.size:LOCAL.size+nlen] != archived_name:
                raise ValueError("Entry offset or name mismatch")
            start = LOCAL.size + nlen + extra
            image = zlib.decompress(data[start:start + compressed], -15)
            if len(image) != raw or zlib.crc32(image) != crc:
                raise ValueError("Entry CRC mismatch")
            with Image.open(io.BytesIO(image)) as source:
                if source.format != "JPEG" or source.size != (512, 512):
                    raise ValueError("Unexpected image format or size")
                source.verify()
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".part")
            tmp.write_bytes(image)
            tmp.replace(target)
            return
        except (OSError, ValueError, struct.error, zlib.error, requests.RequestException) as exc:
            if attempt == 3:
                raise RuntimeError(f"{archived_name.decode()}: {exc}") from exc
            time.sleep(0.5 * 2**attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-cache", type=Path, default=Path("data/VGGFace2-HQ/archive-index.bin"))
    parser.add_argument("--output", type=Path, default=Path("data/VGGFace2-HQ/aligned"))
    parser.add_argument("--limit", type=int, default=4605)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.limit <= 8631 or args.workers < 1:
        parser.error("Use --limit from 1 to 8631 and --workers >= 1.")
    data = read_index(args.index_cache)
    print("Central directory SHA-256", INDEX_SHA256, flush=True)
    items = gallery(data, args.limit)
    pending = []
    for identity, entry in items:
        name, _, _, _, _, raw, crc, _, _ = entry
        target = args.output / identity.decode() / name.decode()
        if not valid_file(target, raw, crc):
            pending.append((target, entry))
    completed = len(items) - len(pending)
    print(f"{completed} verified local files; {len(pending)} to download", flush=True)
    archives = {disk: resolve_archive(disk) for disk in sorted({entry[2] for _, entry in pending})}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch, target, entry, archives[entry[2]]): target.parent.name
            for target, entry in pending
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), completed + 1):
            try:
                future.result()
            except (RuntimeError, ValueError) as exc:
                failures.append(futures[future])
                print(f"Failed {futures[future]}: {exc}", flush=True)
            if index % 100 == 0 or index == len(items):
                print(f"{index}/{len(items)} processed, {len(failures)} failed", flush=True)
    if failures:
        raise SystemExit(f"Failed identities: {len(failures)}")

if __name__ == "__main__":
    main()
