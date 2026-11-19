"""Tests for the ORL dataset loader, plus the opt-in replication on real ORL data."""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from eidos_core.datasets.orl import ensure_orl, iter_orl, load_orl_image
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
