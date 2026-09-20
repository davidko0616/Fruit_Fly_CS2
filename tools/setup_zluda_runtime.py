"""Stage the user-provided ZLUDA build and a pinned AMD SDK inside this repo."""
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local-zluda"
SOURCE = Path(r"C:\Users\chung\Documents\zluda")
SDK_FILE = "therock-dist-windows-gfx110X-all-7.14.0a20260612.tar.gz"
SDK_URL = "https://therock-nightly-tarball.s3.amazonaws.com/" + SDK_FILE
SDK_BYTES = 3052085146


def download_ranges(archive):
    parts = LOCAL / "sdk-download-parts"
    parts.mkdir(exist_ok=True)
    chunk = 16 * 1024 * 1024
    ranges = [(start, min(start + chunk, SDK_BYTES)) for start in range(0, SDK_BYTES, chunk)]

    def fetch(bounds):
        start, end = bounds
        dest = parts / str(start)
        if dest.exists() and dest.stat().st_size == end - start:
            return end - start
        for attempt in range(4):
            try:
                req = urllib.request.Request(SDK_URL, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{SDK_BYTES}":
                        raise RuntimeError("Server did not return the requested byte range")
                    with dest.with_suffix(".partial").open("wb") as out:
                        shutil.copyfileobj(response, out, length=1024 * 1024)
                if dest.with_suffix(".partial").stat().st_size != end - start:
                    raise RuntimeError("Incomplete range")
                dest.with_suffix(".partial").replace(dest)
                return end - start
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    received, last = 0, time.monotonic()
    print("Downloading SDK with 24 resumable connections...", flush=True)
    with ThreadPoolExecutor(max_workers=24) as pool:
        for future in as_completed([pool.submit(fetch, bounds) for bounds in ranges]):
            received += future.result()
            if time.monotonic() - last > 20:
                print(f"SDK: {received / 1e9:.2f}/{SDK_BYTES / 1e9:.2f} GB", flush=True)
                last = time.monotonic()
    with archive.with_suffix(".assembling").open("wb") as out:
        for start, _ in ranges:
            with (parts / str(start)).open("rb") as part:
                shutil.copyfileobj(part, out, length=8 * 1024 * 1024)
    archive.with_suffix(".assembling").replace(archive)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    LOCAL.mkdir(exist_ok=True)
    staged = LOCAL / "zluda"
    staged.mkdir(exist_ok=True)
    # The supplied directory mixes 2024 DLLs with a newer build. Only stage
    # the versioned library names used by the current launcher, plus its helpers.
    files = ["zluda.exe", "nvcuda.dll", "zluda_redirect.dll", "nvml.dll",
             "nvcudart_hybrid64.dll", "cuda_check.exe", "ptxas.exe",
             "zluda_precompile.exe"]
    files += sorted(p.name for p in SOURCE.glob("*64_*.dll"))
    manifest = {"source": str(SOURCE), "sdk_url": SDK_URL, "zluda_files": {}}
    for name in files:
        shutil.copy2(SOURCE / name, staged / name)
        manifest["zluda_files"][name] = sha256(staged / name)
    if (SOURCE / "trace").is_dir():
        shutil.copytree(SOURCE / "trace", staged / "trace", dirs_exist_ok=True)
    archive = LOCAL / SDK_FILE
    if not archive.exists():
        download_ranges(archive)
    if archive.stat().st_size != SDK_BYTES:
        raise RuntimeError("Existing SDK archive has unexpected size")
    manifest["sdk_sha256"] = sha256(archive)
    sdk = LOCAL / "sdk"
    if not (sdk / ".extracted").exists():
        sdk.mkdir(exist_ok=True)
        print("Extracting AMD SDK using Python's safe data filter...", flush=True)
        with tarfile.open(archive) as tar:
            tar.extractall(sdk, filter="data")
        (sdk / ".extracted").write_text(SDK_FILE, encoding="utf-8")
    candidates = list(sdk.rglob("rocblas.dll"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one rocblas.dll, found {candidates}")
    manifest["hip_path"] = str(candidates[0].parent.parent)
    (LOCAL / "runtime.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"hip_path": manifest["hip_path"], "sdk_sha256": manifest["sdk_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
