# Historical ESOL evidence

Only the original fold results, training summary, configuration and normalization
metadata are retained here. These four files were recovered byte-for-byte from Git
HEAD; [checksums.json](checksums.json) records their hashes. Opaque MLflow run
directories, serialized weights and old environment files have been removed from
the working tree; the original tracked artifacts remain in Git history.

This run used two shuffled K-folds, seed 42, and full-dataset target normalization.
Its selected-epoch validation MAEs were 0.557280 and 0.684546 in standardized units,
not five-fold scaffold test MAEs or evidence for a 0.45 logS claim. See the
[compact table](../../results/recorded_esol.csv)
