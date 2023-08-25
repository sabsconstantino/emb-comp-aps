# Comparison of graph and text embedding methods on American Physical Society (APS) data

This repository contains the codebase used for analysis in (paper).

How to use:
1. Ensure that the requirements stated in file `environment.yml` are installed. You may also install the environment directly from file using `conda` (change the environment path as necessary).
2. Run the preprocessing, embedding, and k-nearest neighbors classification pipeline. This is done using the Snakefile (i.e. by running command `snakemake`). Refer to the [Snakemake documentation](https://snakemake.readthedocs.io/) for tips.
3. The evaluation and figures may be executed using the Jupyter notebooks. 
