"""Verify all SHA256 entries in an extracted handoff or its original ZIP."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile


def checksum(stream):
    result = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
        result.update(chunk)
        size += len(chunk)
    return result.hexdigest(), size


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifests = [name for name in names if name.endswith("/manifest.json") and name.count("/") == 1]
        if len(manifests) != 1:
            raise AssertionError("Exactly one package-root manifest required")
        manifest_name = manifests[0]
        prefix = manifest_name[:-len("manifest.json")]
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        expected_names = {manifest_name} | {prefix + item["archive_path"] for item in manifest["files"]}
        if set(names) != expected_names or len(names) != len(expected_names):
            raise AssertionError("Missing, duplicate or unmanifested ZIP entries")
        total = 0
        for item in manifest["files"]:
            relative = PurePosixPath(item["archive_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise AssertionError("Unsafe archive path")
            with archive.open(prefix + item["archive_path"]) as stream:
                actual, size = checksum(stream)
            if actual != item["sha256"] or size != item["bytes"]:
                raise AssertionError("Hash or size differs: " + item["archive_path"])
            total += size
        return {"status": "passed", "files_verified": len(manifest["files"]), "uncompressed_bytes_verified": total,
                "platform_feedback_records": manifest["summary"]["platform_feedback_records"],
                "submission_files": manifest["summary"]["submission_files"]}


def verify_directory(path):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    total = 0
    for item in manifest["files"]:
        candidate = (root / item["archive_path"]).resolve()
        if root != candidate and root not in candidate.parents:
            raise AssertionError("Unsafe relative path")
        with candidate.open("rb") as stream:
            actual, size = checksum(stream)
        if actual != item["sha256"] or size != item["bytes"]:
            raise AssertionError("Hash or size differs: " + item["archive_path"])
        total += size
    return {"status": "passed", "files_verified": len(manifest["files"]), "uncompressed_bytes_verified": total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = verify_zip(args.zip) if args.zip else verify_directory(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
