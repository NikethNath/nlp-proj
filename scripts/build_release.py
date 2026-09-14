"""Bundle the generated corpus into dist/ for a versioned release.

    PYTHONPATH=src .venv/bin/python scripts/build_release.py v0.1

Produces dist/orthography-corpus-<version>.tar.gz containing the FLORES and
DravidianCodeMix JSONL files, the human-collection manifest, the data card,
corpus statistics and SHA-256 checksums, plus any ingested human responses.
Publishing is a separate, deliberate step (the command is printed at the end).
"""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"

EXPECTED = [f"flores_{l}_{s}.jsonl" for l in ("ml", "ta") for s in ("dev", "devtest")] + \
           [f"dcm_{l}_{s}.jsonl" for l in ("ml", "ta") for s in ("train", "dev", "test")]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: build_release.py <version>, e.g. v0.1")
    version = sys.argv[1]
    missing = [name for name in EXPECTED if not (DERIVED / name).exists()]
    if missing:
        sys.exit(f"missing generated files {missing}; build the corpus first (see README)")

    files = {f"data/{name}": DERIVED / name for name in EXPECTED}
    for path in sorted((DERIVED / "human").glob("responses_*.jsonl")):
        files[f"data/human/{path.name}"] = path
    files["README.md"] = ROOT / "docs" / "datacard.md"
    files["corpus_stats.json"] = ROOT / "docs" / "corpus_stats.json"
    files["collection_manifest.json"] = ROOT / "collection" / "manifest.json"

    sums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {name}\n"
                   for name, p in sorted(files.items()))
    out = ROOT / "dist" / f"orthography-corpus-{version}.tar.gz"
    out.parent.mkdir(exist_ok=True)
    prefix = f"orthography-corpus-{version}"
    with tarfile.open(out, "w:gz") as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=f"{prefix}/{name}")
        sums_path = out.parent / "SHA256SUMS"
        sums_path.write_text(sums)
        tar.add(sums_path, arcname=f"{prefix}/SHA256SUMS")
        sums_path.unlink()

    stats = json.loads((ROOT / "docs" / "corpus_stats.json").read_text())
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files)")
    print(f"FLORES sentences per language: {stats['flores']['ml']['sentences']}")
    print("Publish with:")
    print(f"  gh release create corpus-{version} {out} --title 'Corpus {version}' "
          "--notes-file docs/datacard.md")


if __name__ == "__main__":
    main()
