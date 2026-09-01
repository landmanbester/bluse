import numpy as np

from tests.unit import fixtures


def test_tie_column_has_the_planted_tie():
    X, cols, _ = fixtures.synthetic_matrix(n=500, seed=0)
    tie = X[:, cols.index("tie_col")]
    v, c = np.unique(tie, return_counts=True)
    assert c.max() / len(tie) == 0.30
    assert v[c.argmax()] == 0.0


def test_ordinal_column_has_the_planted_level_count():
    X, cols, _ = fixtures.synthetic_matrix(n=500, seed=0)
    assert len(np.unique(X[:, cols.index("ordinal_col")])) == 8


def test_labelled_fixture_has_the_planted_narrow_share():
    labels, df = fixtures.synthetic_labelled(n=600, seed=0)
    narrow = 0
    for c in np.unique(labels):
        f = df.frequency[labels == c]
        if f.max() - f.min() < 1.0:
            narrow += int((labels == c).sum())
    assert narrow / len(labels) == 0.20


def test_centroid_fixture_has_three_separated_families():
    labels, X = fixtures.synthetic_centroid_space(seed=0)
    assert len(np.unique(labels)) == 9
    cent = np.array([X[labels == c].mean(0) for c in np.unique(labels)])
    assert np.linalg.norm(cent[0] - cent[1]) < 2.0
    assert np.linalg.norm(cent[0] - cent[3]) > 8.0
