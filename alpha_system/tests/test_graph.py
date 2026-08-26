from __future__ import annotations

import numpy as np

from src.data.graph import build_relation_matrix


def test_relation_matrix_structure(synth_arrays):
    close = synth_arrays["close_adj"]
    rel = build_relation_matrix(close, 10, 260, corr_window=250, threshold=0.9)
    n = close.shape[1]
    assert rel.shape == (n, n, 2)
    assert set(np.unique(rel)) <= {0.0, 1.0}
    assert np.all(np.diagonal(rel[:, :, 0]) == 1.0)
    assert np.all(np.diagonal(rel[:, :, 1]) == 1.0)
    assert np.allclose(rel[:, :, 0], rel[:, :, 0].T)


def test_high_threshold_gives_sparse_graph(synth_arrays):
    close = synth_arrays["close_adj"]
    rel = build_relation_matrix(close, 10, 260, corr_window=250, threshold=0.995)
    off_diag_edges = rel[:, :, 0].sum() - close.shape[1]
    assert off_diag_edges < close.shape[1] ** 2 / 4
