"""Tests for the ORL dataset loader, plus the opt-in replication on real ORL data."""

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from eidos_core.datasets.orl import ensure_orl, iter_orl, load_orl_image
from eidos_core.detection import detect
from eidos_core.eigenfaces import project, train
from eidos_core.preprocess import prepare
from tests.orl_fake import write_fake_orl


def test_iter_orl_reads_fake_tree(tmp_path: Path) -> None:
    write_fake_orl(tmp_path, subjects=2, per_subject=2)
    items = list(iter_orl(tmp_path))
    assert [(s, i) for s, i, _ in items] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert all(img.shape == (112, 92) and img.dtype == np.uint8 for _, _, img in items)
    np.testing.assert_array_equal(load_orl_image(tmp_path, 2, 1), items[2][2])


def test_ensure_orl_from_local_zip(tmp_path: Path) -> None:
    source = write_fake_orl(tmp_path / "src", subjects=1, per_subject=1)
    archive = tmp_path / "att_faces.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(source / "s1" / "1.pgm", "s1/1.pgm")

    dest = tmp_path / "orl"
    assert ensure_orl(dest, url=archive.as_uri()) == dest
    assert (dest / "s1" / "1.pgm").exists()

    archive.unlink()  # a second call must not download again
    assert ensure_orl(dest, url=archive.as_uri()) == dest


@pytest.mark.dataset
def test_orl_replication_accuracy(tmp_path_factory: pytest.TempPathFactory) -> None:
    root = ensure_orl(tmp_path_factory.mktemp("orl"))
    full = (0, 0, 92, 112)
    train_x, train_y, test_x, test_y = [], [], [], []
    for subject, index, image in iter_orl(root):
        if subject > 30:
            continue
        vector = prepare(image, full)
        if index <= 6:
            train_x.append(vector)
            train_y.append(subject)
        else:
            test_x.append(vector)
            test_y.append(subject)

    model = train(np.array(train_x), train_y, n_components=20)
    correct = 0
    for face, label in zip(test_x, test_y, strict=True):
        distances = np.linalg.norm(model.class_vectors - project(model, face), axis=1)
        correct += int(model.class_ids[np.argmin(distances)] == label)
    accuracy = correct / len(test_y)
    print(f"ORL accuracy at M'=20: {accuracy:.3f}")
    assert accuracy >= 0.85


def test_ensure_orl_rejects_archive_without_orl_photos(tmp_path: Path) -> None:
    archive = tmp_path / "other.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("README", "not the ORL database")
    with pytest.raises(ValueError, match="no ORL photos"):
        ensure_orl(tmp_path / "orl", url=archive.as_uri())
    assert not (tmp_path / "orl").exists()


def _portrait_hits(face_fraction: float, tmp_path_factory: pytest.TempPathFactory) -> int:
    """Detection hits (box overlaps the true face by more than half) for 14 held-out
    ORL faces placed in a 600 x 800 grey portrait at `face_fraction` of its width."""
    root = ensure_orl(tmp_path_factory.getbasetemp() / "orl")
    faces, labels = [], []
    for subject, index, image in iter_orl(root):
        if index <= 8:
            faces.append(prepare(image, (0, 0, 92, 112), standardize_face=True))
            labels.append(subject)
    model = train(np.array(faces), labels, n_components=20)

    hits = 0
    for subject in range(1, 41, 3):
        photo = np.full((800, 600), 128.0)
        h = round(face_fraction * 600)
        w = round(h * 92 / 112)
        x, y = (600 - w) // 2, (800 - h) // 3
        face = load_orl_image(root, subject, 10).astype(np.float64)
        photo[y : y + h, x : x + w] = cv2.resize(face, (w, h), interpolation=cv2.INTER_CUBIC)
        bx, by, bw, bh = detect(model, photo, normalize=True).box
        overlap = max(0, min(bx + bw, x + w) - max(bx, x)) * max(
            0, min(by + bh, y + h) - max(by, y)
        )
        hits += overlap / (bw * bh + w * h - overlap) > 0.5
    return hits


@pytest.mark.dataset
def test_detection_finds_medium_portrait_faces(tmp_path_factory: pytest.TempPathFactory) -> None:
    assert _portrait_hits(0.42, tmp_path_factory) >= 12


@pytest.mark.dataset
@pytest.mark.xfail(
    strict=True,
    reason="Known limitation (1.1.0): when a real face fills most of the photo, smaller "
    "boxes on part of the face score as more face-like than the whole face (ε is not "
    "comparable across sizes). Measured: 4/14 hits at 70%. If this passes, update the docs.",
)
def test_detection_finds_large_portrait_faces(tmp_path_factory: pytest.TempPathFactory) -> None:
    assert _portrait_hits(0.70, tmp_path_factory) >= 12
