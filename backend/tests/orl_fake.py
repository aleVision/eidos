"""Write a tiny fake ORL tree (s<subject>/<index>.pgm) so tests never need the network."""

from pathlib import Path

import numpy as np
from PIL import Image


def write_fake_orl(root: Path, subjects: int, per_subject: int) -> Path:
    rng = np.random.default_rng(3)
    for s in range(1, subjects + 1):
        prototype = rng.uniform(0, 255, (112, 92))
        folder = root / f"s{s}"
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, per_subject + 1):
            pixels = np.clip(prototype + rng.normal(0, 4, prototype.shape), 0, 255)
            Image.fromarray(pixels.astype(np.uint8)).save(folder / f"{i}.pgm")
    return root
