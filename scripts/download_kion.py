"""Download and extract the KION files used by the project."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/irsafilo/KION_DATASET/main"
OUTPUT_DIR = Path("downloads/kion")
ARCHIVE = "data_original.zip"


def download_file(filename: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[skip] {destination} already exists")
        return destination

    partial = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(
        f"{BASE_URL}/{filename}",
        headers={"User-Agent": "GPTRec-base/0.1"},
    )

    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    print(f"[downloaded] {destination}")
    return destination


def extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    safe_root = destination.resolve()

    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(safe_root):
                raise ValueError(f"Unsafe path in {archive.name}: {member.filename}")

        for member in zip_file.infolist():
            target = destination / member.filename
            if not target.exists():
                zip_file.extract(member, destination)

    print(f"[extracted] {archive.name} -> {destination}")


def main() -> None:
    archives_dir = OUTPUT_DIR / "archives"
    raw_dir = OUTPUT_DIR / "raw"

    archive = download_file(ARCHIVE, archives_dir / ARCHIVE)
    extract_zip(archive, raw_dir)


if __name__ == "__main__":
    main()

