# ESOL data provenance

`ESOL/esol/raw/delaney-processed.csv` is the MoleculeNet/DeepChem distribution
of the Delaney ESOL dataset (1,128 rows). It is retained as a small input fixture
so the research workflow can regenerate molecular graphs without downloading data.

- Source: [DeepChem ESOL CSV](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv)
- Loader: [PyG MoleculeNet 2.6.1](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.MoleculeNet.html)
- Dataset paper: John S. Delaney, *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure* (2004), [DOI](https://doi.org/10.1021/ci034243x)
- SHA-256: `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189`

The CSV is third-party data; the project MIT license applies to project code and
does not relicense this dataset. Preserve its source attribution when redistributing.

PyG regenerates ignored `ESOL/esol/processed/*.pt` caches from this input.
Do not commit those caches. Keep the original row order when reconstructing the
historical shuffled K-fold indices. Remove stale local processed caches if the
raw data or processing dependencies change.
