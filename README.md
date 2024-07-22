# Analysis Workflow and Methodology

To align with the original authors' methods, we conducted a comprehensive inspection of parameter grids for both t-SNE and UMAP. Since the authors did not provide code or parameter references for t-SNE, we performed the following steps across several Jupyter notebooks to replicate and extend their analysis for murine samples:

## Data Pooling

- **pooling_samples.ipynb**:
  - Combined samples in various configurations (all samples, all murine spleen, all murine blood).
  - Exported the pooled datasets as `.h5ad` files to the `/data/pooled_data` folder, streamlining sample loading and avoiding repetitive pooling in subsequent analyses.

## Comprehensive Analysis of Pooled Samples

- **all_mouse_samples.ipynb**:
  - Loaded pooled murine samples.
  - Performed t-SNE and UMAP analyses to explore data embedding and visualization.
  - Conducted tissue-specific marker gene detection for spleen and blood tissues.

## Individual Sample Analysis

- **spleen_individual_inspection.ipynb** / **blood_individual_inspection.ipynb**:
  - Analyzed each individual mouse spleen and blood donor separately.
  - Explored the effects of different t-SNE and UMAP parameters on data embedding and visualization.

## Pooled Sample Analysis

- **spleen_pooled_inspection.ipynb** / **blood_pooled_inspection.ipynb**:
  - Focused on pooled spleen and blood samples.
  - Determined optimal t-SNE and UMAP parameters for effective visualization.
  - Applied clustering algorithms (k-means, Gaussian Mixture Model (GMM), and hierarchical clustering) to identify clusters and compare with the number reported by the authors.
  - Investigated driving genes of each cluster for comparison with the authors' results, noting that the absence of provided cluster labels limited direct biological validation.

## Machine Learning Classification

- **machine_learning_pipeline.ipynb**:
  - Implemented different classifiers to distinguish cells from spleen or blood tissue.
  - Evaluated classifier performance and selected the most effective models based on accuracy and computational efficiency.

