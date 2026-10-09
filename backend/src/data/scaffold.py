"""Deterministic nested scaffold folds, independent of targets."""

from collections import defaultdict
import random

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SCAFFOLD_POLICY = "bemis-murcko-no-chirality-acyclic-together-v1"


def scaffold_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid molecule for scaffold splitting: {smiles!r}")
    # ALL acyclic molecules have the empty scaffold and stay in one group.
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_folds(smiles, n_folds=5, val_fraction=0.1, seed=42):
    """Outer held-out test folds plus a scaffold-disjoint inner validation set.

    Largest groups are greedily placed into the smallest outer fold; seeded
    shuffling breaks size ties. Validation groups approximate the requested
    fraction without splitting any scaffold. Exact row and scaffold IDs are saved.
    """
    if n_folds < 2 or not 0 < val_fraction < 1:
        raise ValueError("Require n_folds >= 2 and 0 < val_fraction < 1")
    keys = [scaffold_key(s) for s in smiles]
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    if len(groups) < n_folds + 1:
        raise ValueError("Need at least n_folds + 1 distinct scaffolds for nested folds")
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    outer = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for key, rows in items:
        fold = min(range(n_folds), key=lambda i: (sizes[i], i))
        outer[fold].append(key)
        sizes[fold] += len(rows)
    folds = []
    for i, test_keys in enumerate(outer):
        remaining = [key for key in groups if key not in set(test_keys)]
        if len(remaining) < 2:
            raise ValueError("Not enough training scaffolds to form an inner validation set")
        random.Random(seed + i + 1).shuffle(remaining)
        target = val_fraction * sum(len(groups[key]) for key in remaining)
        val_keys = []
        count = 0
        # Always leave at least one group for training.
        for key in remaining[:-1]:
            if not val_keys or abs(count + len(groups[key]) - target) < abs(count - target):
                val_keys.append(key)
                count += len(groups[key])
        train_keys = [key for key in remaining if key not in set(val_keys)]
        fold = {"fold_idx": i}
        for label, selected in [("train", train_keys), ("validation", val_keys), ("test", test_keys)]:
            fold[label + "_indices"] = sorted(row for key in selected for row in groups[key])
        folds.append(fold)
    manifest = {"split_type": "scaffold", "policy": SCAFFOLD_POLICY, "seed": seed,
                "n_folds": n_folds, "validation_fraction_of_outer_training": val_fraction,
                "smiles": list(smiles), "scaffold_ids": keys, "folds": folds}
    validate_scaffold_manifest(manifest)
    return manifest


def validate_scaffold_manifest(manifest):
    keys, n = manifest["scaffold_ids"], len(manifest["smiles"])
    if len(keys) != n or keys != [scaffold_key(s) for s in manifest["smiles"]]:
        raise ValueError("Scaffold IDs do not match molecule identifiers")
    tested = []
    for fold in manifest["folds"]:
        rows = [fold[label + "_indices"] for label in ("train", "validation", "test")]
        if any(not part for part in rows):
            raise ValueError("All three partitions must be nonempty")
        if sorted(row for part in rows for row in part) != list(range(n)):
            raise ValueError("Partitions must cover each molecule exactly once")
        scaffold_sets = [{keys[row] for row in part} for part in rows]
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            if scaffold_sets[a] & scaffold_sets[b]:
                raise ValueError("Scaffold leakage between partitions")
        tested.extend(rows[2])
    if sorted(tested) != list(range(n)):
        raise ValueError("Outer test folds must cover each molecule exactly once")
