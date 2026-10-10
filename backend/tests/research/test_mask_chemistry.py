"""Explanation diagnostics have explicit chemistry and intervention semantics."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.masks import auc, chemical_atoms, clustered_interval, diagnose, signed_mask


def test_chemical_proxies_use_atom_order_and_exclude_polar_adjacent_carbon():
    mol, polar, nonpolar, logp = chemical_atoms("CCO")
    assert [a.GetSymbol() for a in mol.GetAtoms()] == ["C", "C", "O"]
    assert polar.tolist() == [False, False, True]
    assert nonpolar.tolist() == [True, False, False]
    assert len(logp) == 3
    assert auc([0, 100, 1], polar, nonpolar) == 1
    assert auc([1, 100, 1], polar, nonpolar) == 0.5
    assert auc([0, 0], np.array([False, False]), np.array([True, True])) is None


def test_readout_interventions_preserve_graph_and_recover_known_signed_contributions():
    masks = np.array([[.9, .1], [.8, .2], [.1, .9]])
    h = np.array([[1.], [2.], [3.]])
    pooled = (masks[:, :, None] * h[:, None, :]).reshape(3, -1).sum(0)
    def readout(values):
        return values @ np.array([-1., 1.]) + 7
    kwargs = dict(smiles="CCO", nodes=masks, random_nodes=masks[::-1], embeddings=h,
                  pooled=pooled, readout=readout, scale=2, signs=[-1, 1], repeats=20)
    row, atoms = diagnose(**kwargs, rng=np.random.default_rng(42))
    again, _ = diagnose(**kwargs, rng=np.random.default_rng(42))
    assert row == again
    np.testing.assert_allclose(atoms["signed_atom_readout_effect_logS"], [-1.6, -2.4, 4.8])
    assert row["mask_chemical_auc"] == row["signed_atom_effect_chemical_auc"] == 1
    assert row["deleted_atom_count"] == 1
    assert row["polar_atoms"] == row["nonpolar_atoms"] == 1
    np.testing.assert_array_equal(masks, [[.9, .1], [.8, .2], [.1, .9]])


def test_mask_score_is_invariant_to_channel_wide_scale():
    nodes = np.array([[.9, .1], [.1, .9]])
    np.testing.assert_allclose(signed_mask(nodes, [-1, 1]), signed_mask(nodes * [5, 2], [-1, 1]))


def test_bootstrap_keeps_scaffold_clusters_and_molecule_weighted_mean():
    rows = [{"scaffold": "", "score": 1}, {"scaffold": "", "score": 1},
            {"scaffold": "ring", "score": 0}, {"scaffold": "missing", "score": None}]
    record = clustered_interval(rows, "score", np.random.default_rng(42), repeats=200)
    assert record["mean"] == 2 / 3
    assert record["molecule_count"] == 3 and record["scaffold_count"] == 2
    assert record["ci95"] == [0, 1]
