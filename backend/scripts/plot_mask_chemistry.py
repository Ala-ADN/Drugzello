"""Plots for the recorded mask-chemistry experiment; no model execution."""

import csv
import gzip
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/mask-chemistry"
COHORTS = ["paper_random_prediction_only", "scaffold_prediction_only", "scaffold_joint_auxiliary"]
LABELS = ["Original MEGAN\nrandom folds", "MEGAN, prediction only\nscaffold folds", "MEGAN, auxiliary losses\nscaffold folds"]


def main():
    summaries = [json.loads((BASE / c / "summary.json").read_text()) for c in COHORTS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    y = np.arange(3)
    for field, offset, color, label in [("mask_chemical_auc", -.12, "#2166ac", "Learned masks"),
                                        ("random_weights_chemical_auc", .12, "#888888", "Randomized weights")]:
        records = [s["metric_intervals"][field] for s in summaries]
        values = np.array([r["mean"] for r in records])
        errors = np.array([[r["mean"] - r["ci95"][0], r["ci95"][1] - r["mean"]] for r in records]).T
        axes[0].errorbar(values, y+offset, xerr=errors, fmt="o", color=color, capsize=3, label=label)
    axes[0].axvline(.5, color="black", linestyle="--", linewidth=1, label="Shuffled-mask expectation")
    axes[0].set(xlabel="Mean within-molecule chemical-proxy AUC", title="Chemical agreement", xlim=(.3, .95))
    records = [s["metric_intervals"]["top_minus_random_deletion_logS"] for s in summaries]
    values = np.array([r["mean"] for r in records])
    errors = np.array([[r["mean"] - r["ci95"][0], r["ci95"][1] - r["mean"]] for r in records]).T
    axes[1].errorbar(values, y, xerr=errors, fmt="o", color="#2166ac", capsize=3)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Extra absolute prediction change over random (logS)", title="Top-quarter mask readout deletion", xlim=(-.03, .75))
    for ax in axes:
        ax.set_yticks(y, LABELS); ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.20), ncol=3, fontsize=8)
    fig.suptitle("Frozen out-of-fold models; 95% scaffold-cluster bootstrap intervals", fontsize=11)
    fig.savefig(BASE / "chemistry_summary.png", dpi=180, bbox_inches="tight")
    fig.savefig(BASE / "chemistry_summary.svg", bbox_inches="tight")
    plt.close(fig)
    masks, rows = [], []
    for cohort in COHORTS:
        with gzip.open(BASE / cohort / "atom_masks.json.gz", "rt") as f:
            masks.append({r["row_idx"]: r for r in json.load(f)})
        rows.append({int(r["row_idx"]): r for r in csv.DictReader((BASE / cohort / "molecule_metrics.csv").open())})
    # Select by structure and row order, never by attribution quality or error.
    chosen = {}
    for j in sorted(set.intersection(*[set(m) for m in masks])):
        a = masks[0][j]; n = len(a["symbols"])
        if not 6 <= n <= 12 or not any(a["polar"]) or not any(a["nonpolar"]) or "." in a["smiles"]:
            continue
        mol = Chem.MolFromSmiles(a["smiles"])
        kind = "aromatic" if any(atom.GetIsAromatic() for atom in mol.GetAtoms()) else "nonaromatic"
        if kind not in chosen:
            chosen[kind] = j
        if len(chosen) == 2:
            break
    selected = list(chosen.values())
    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 6.2), constrained_layout=True)
    for r, j in enumerate(selected):
        for c, cohort in enumerate(COHORTS):
            atom = masks[c][j]
            mol = Chem.MolFromSmiles(atom["smiles"])
            score = np.array(atom["signed_mask_score"])
            color_values = .5 + .5 * score / max(abs(score).max(), 1e-12)
            colors = {i: tuple(plt.get_cmap("RdBu")(v)[:3]) for i, v in enumerate(color_values)}
            drawing = rdMolDraw2D.MolDraw2DCairo(440, 260)
            drawing.drawOptions().addAtomIndices = True
            drawing.DrawMolecule(mol, highlightAtoms=list(range(len(score))), highlightAtomColors=colors, highlightBonds=[])
            drawing.FinishDrawing()
            axes[r, c].imshow(Image.open(io.BytesIO(drawing.GetDrawingText())))
            axes[r, c].axis("off")
            pred = float(rows[c][j]["predicted_logS"])
            axes[r, c].set_title(f"{LABELS[c]}\nRow {j}; prediction {pred:.2f} logS", fontsize=9)
    fig.suptitle("Training-oriented mask score: blue positive, red negative; intensity scaled per molecule\nExamples selected by size, chemistry and row order, without inspecting masks", fontsize=10)
    fig.savefig(BASE / "molecule_examples.png", dpi=180)
    plt.close(fig)
    (BASE / "figure_examples.json").write_text(json.dumps({"row_indices": selected, "selection": "first eligible nonaromatic and aromatic rows; 6–12 atoms, both proxies, no salts"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
