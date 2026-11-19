"""The ORL (AT&T) Database of Faces: 40 people x 10 grey photos of 92 x 112 pixels.

Used by Eidos for demos and to replicate the paper's experiment. It is only
downloaded when explicitly requested and is never committed to the repository.
When showing these images, credit ORL_CREDIT.
"""

import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

ORL_URL = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"
ORL_CREDIT = "Olivetti Research Laboratory (AT&T Laboratories Cambridge)"

# Only extract files shaped like s<subject>/<index>.pgm (also blocks path traversal).
_MEMBER = re.compile(r"^s(\d+)/(\d+)\.pgm$")


def ensure_orl(dest: Path, url: str = ORL_URL) -> Path:
    """Download and extract ORL into `dest` once; return `dest` (containing s1..s40)."""
    if (dest / "s1" / "1.pgm").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archive = Path(tmp) / "orl.zip"
        with urllib.request.urlopen(url) as response, archive.open("wb") as file:
            shutil.copyfileobj(response, file)
        staging = Path(tmp) / "extracted"
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if _MEMBER.match(member):
                    z.extract(member, staging)
        shutil.rmtree(dest, ignore_errors=True)
        staging.rename(dest)
    return dest


def load_orl_image(root: Path, subject: int, index: int) -> np.ndarray:
    """One ORL photo as a uint8 array of shape (112, 92)."""
    with Image.open(root / f"s{subject}" / f"{index}.pgm") as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def iter_orl(root: Path) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield (subject, index, image) for every photo, sorted numerically."""
    pairs = []
    for path in root.glob("s*/*.pgm"):
        match = _MEMBER.match(f"{path.parent.name}/{path.name}")
        if match:
            pairs.append((int(match.group(1)), int(match.group(2))))
    for subject, index in sorted(pairs):
        yield subject, index, load_orl_image(root, subject, index)
