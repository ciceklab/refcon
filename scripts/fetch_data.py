#!/usr/bin/env python3
"""Fetch REFCON's data and model checkpoints from Zenodo into ./data and ./checkpoints.

The processed evaluation cohorts (expression + copy-number ground truth) and the model checkpoints are
hosted on Zenodo (doi:10.5281/zenodo.23534511; they are too large to keep in git). Run this once after cloning.

  python scripts/fetch_data.py

Archives
  data.zip         data/<cohort>/...  (one subdirectory per evaluation cohort)
  checkpoints.zip  checkpoints/refcon_v3_s2.pt, checkpoints/refcon_v3_s3.pt
An archive is skipped when every file it provides is already present. Each download is checked against its SHA-256.
"""
import hashlib, urllib.request, zipfile, tempfile, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZENODO_RECORD = "10.5281/zenodo.23534511"

# archive -> (folder, files it provides inside that folder, SHA-256 of the archive)
DATA_FILES = [f"{c}/{s}" for c, stems in {
    "a375": ["a375_dntr_per_cell"], "hct116": ["hct116_dntr_per_cell"], "scone": ["scone_astrocytoma"],
    "ccle_esophageal": ["ccle_esophageal"], "ccle_mixed": ["ccle_mixed"], "mixseq": ["mixseq_pancancer"]}.items()
    for stem in stems for s in (f"{stem}.h5ad", f"{stem}_cn.npy")]
ARCHIVES = {
    "data.zip": ("data", DATA_FILES, "3b9be7fa7d134342af046010dc56d7097a906b40357348954b8da36103029025"),
    "checkpoints.zip": ("checkpoints", ["refcon_v3_s2.pt", "refcon_v3_s3.pt"],
                        "19f4d5b309b8a4ef6806f48e10f5e034fcb51cd7d7c896a3425b761d84f2f3fe"),
}


def present(dst, files):
    # data/README.md is in git, so a folder that is merely non-empty does not mean the archive was fetched
    return all((ROOT / dst / f).exists() for f in files)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    base = f"https://zenodo.org/records/{ZENODO_RECORD.split('.')[-1]}/files"
    for fn, (dst, files, digest) in ARCHIVES.items():
        if present(dst, files):
            print(f"[skip ] {fn}: already present in {dst}/")
            continue
        url = f"{base}/{fn}?download=1"
        print(f"[get  ] {url}")
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.close()
        try:
            urllib.request.urlretrieve(url, tmp.name)
            if sha256(tmp.name) != digest:
                raise SystemExit(f"{fn}: SHA-256 does not match the Zenodo record; download it again")
            print(f"[unzip] {fn} -> {dst}/")
            with zipfile.ZipFile(tmp.name) as z:
                z.extractall(ROOT)
        finally:
            os.unlink(tmp.name)
    print("done.")


if __name__ == "__main__":
    main()
