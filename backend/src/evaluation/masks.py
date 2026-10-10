"""Framework-independent chemical-proxy and readout-intervention diagnostics."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chemical_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    polar = np.array([a.GetAtomicNum() in (7, 8) for a in mol.GetAtoms()])
    nonpolar = np.array([a.GetAtomicNum() == 6 and a.GetFormalCharge() == 0
                        and all(n.GetAtomicNum() == 6 for n in a.GetNeighbors()) for a in mol.GetAtoms()])
    logp = np.array([c[0] for c in rdMolDescriptors._CalcCrippenContribs(mol)])
    return mol, polar, nonpolar, logp


def auc(scores, polar, nonpolar):
    if not polar.any() or not nonpolar.any():
        return None
    keep = polar | nonpolar
    ranks = rankdata(np.asarray(scores)[keep])
    positive = polar[keep]
    p, n = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - p * (p + 1) / 2) / (p * n))


def spearman(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def signed_mask(nodes, signs):
    # Remove channel-wide scale so one dense channel cannot dominate by magnitude.
    return (nodes / np.maximum(nodes.mean(0), 1e-12)) @ np.asarray(signs)


def diagnose(smiles, nodes, random_nodes, embeddings, pooled, readout, scale, signs, rng, repeats=100):
    """Readout accepts a batch of pooled vectors and returns standardized outputs."""
    mol, polar, nonpolar, logp = chemical_atoms(smiles)
    n, k = nodes.shape
    assert mol.GetNumAtoms() == n == len(embeddings) == len(random_nodes)
    assert np.isfinite(nodes).all() and np.isfinite(embeddings).all()
    contribution = (nodes[:, :, None] * embeddings[:, None, :]).reshape(n, -1)
    np.testing.assert_allclose(contribution.sum(0), pooled, atol=3e-5, rtol=2e-5)
    base = float(readout(pooled[None])[0])
    atom_delta = (base - readout(pooled[None] - contribution)) * scale
    score = signed_mask(nodes, signs)
    random_score = signed_mask(random_nodes, signs)
    magnitude = nodes.mean(1)
    count = max(1, int(np.ceil(n * 0.25)))
    # Seeded tie breaking avoids interpreting atom-index order as an explanation.
    tie_order = rng.permutation(n)
    order = tie_order[np.argsort(magnitude[tie_order], kind="stable")]
    selected = [order[-count:], order[:count]]
    selected += [rng.choice(n, count, replace=False) for _ in range(repeats)]
    removed = np.stack([contribution[ids].sum(0) for ids in selected])
    delta = np.abs(base - readout(pooled[None] - removed)) * scale
    shuffled = np.stack([score[rng.permutation(n)] for _ in range(repeats)])
    null_auc = [auc(s, polar, nonpolar) for s in shuffled]
    chemical_auc = auc(score, polar, nonpolar)
    random_auc = auc(random_score, polar, nonpolar)
    row = {
        "atom_count": n, "polar_atoms": int(polar.sum()), "nonpolar_atoms": int(nonpolar.sum()),
        "deleted_atom_count": count,
        "mask_chemical_auc": chemical_auc,
        "random_weights_chemical_auc": random_auc,
        "mask_minus_random_weights_auc": chemical_auc - random_auc if chemical_auc is not None else None,
        "permuted_mask_chemical_auc": float(np.mean(null_auc)) if chemical_auc is not None else None,
        "mask_vs_negative_atomic_logp_spearman": spearman(score, -logp),
        "signed_atom_effect_chemical_auc": auc(atom_delta, polar, nonpolar),
        "signed_atom_effect_vs_negative_atomic_logp_spearman": spearman(atom_delta, -logp),
        "mask_vs_absolute_atom_effect_spearman": spearman(magnitude, abs(atom_delta)),
        "trained_vs_random_mask_spearman": spearman(magnitude, random_nodes.mean(1)),
        "top_mask_deletion_abs_logS": float(delta[0]),
        "bottom_mask_deletion_abs_logS": float(delta[1]),
        "random_deletion_abs_logS": float(delta[2:].mean()),
        "top_minus_random_deletion_logS": float(delta[0] - delta[2:].mean()),
        "top_minus_bottom_deletion_logS": float(delta[0] - delta[1]),
        "fixed_channel_polarity_chemical_auc": auc(signed_mask(nodes, [-1, 1]), polar, nonpolar) if k == 2 else None,
    }
    atoms = {"symbols": [a.GetSymbol() for a in mol.GetAtoms()], "polar": polar.tolist(),
             "nonpolar": nonpolar.tolist(), "atomic_logp": logp.tolist(),
             "masks": nodes.tolist(), "parameter_randomized_masks": random_nodes.tolist(),
             "signed_mask_score": score.tolist(), "signed_atom_readout_effect_logS": atom_delta.tolist()}
    return row, atoms


def clustered_interval(rows, field, rng, repeats=2000):
    groups = {}
    for row in rows:
        value = row.get(field)
        if value is not None and np.isfinite(value):
            groups.setdefault(row["scaffold"], []).append(value)
    if not groups:
        return None
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    ids = rng.integers(0, len(groups), size=(repeats, len(groups)))
    means = sums[ids].sum(1) / sizes[ids].sum(1)
    return {"mean": float(sums.sum() / sizes.sum()), "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "molecule_count": int(sizes.sum()), "scaffold_count": len(groups)}


def save_results(out, cohort, rows, atom_records, fold_records, evidence):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert len({r["row_idx"] for r in rows}) == len(rows)
    for row in rows:
        row["scaffold"] = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(row["smiles"]), includeChirality=False)
    metrics = list(rows[0])[5:]
    metrics = [m for m in metrics if m != "scaffold"]
    rng = np.random.default_rng(20261010)
    summary = {"cohort": cohort, "molecules": len(rows), "folds": fold_records,
               "metric_intervals": {m: clustered_interval(rows, m, rng) for m in metrics},
               "bootstrap_unit": "scaffold clusters", "bootstrap_repeats": 2000}
    for name, value in [("summary.json", summary), ("evidence.json", evidence)]:
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    with (out / "molecule_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with gzip.GzipFile(filename=str(out / "atom_masks.json.gz"), mode="wb", mtime=0) as f:
        f.write(json.dumps(atom_records, separators=(",", ":"), allow_nan=False).encode())
    print(json.dumps({"cohort": cohort, "molecules": len(rows), "metrics": summary["metric_intervals"]}, indent=2), flush=True)
