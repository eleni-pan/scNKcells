# Analysis Workflow and Methodology

This project attempts to replicate the single cell RNA sequencing analysis done by Crinier et al. in the paper High-Dimensional Single-Cell Analysis Identifies Organ-Specific Signatures and Conserved NK Cell Subsets in Humans and Mice. 

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
  - Analyzed each individual spleen and blood mouse donor separately.
  - Explored the effects of different t-SNE and UMAP parameters on data embedding and visualization.

## Pooled Sample Analysis

- **spleen_pooled_inspection.ipynb** / **blood_pooled_inspection.ipynb**:
  - Focused on pooled spleen and blood samples.
  - Determined optimal t-SNE and UMAP parameters for effective visualization.
  - Applied clustering algorithms (k-means, Gaussian Mixture Model (GMM), and hierarchical clustering) to identify clusters and compare with the number reported by the authors.
 
  
## Machine Learning Classification

- **machine_learning_pipeline.ipynb**:
  - Implemented different classifiers to distinguish cells from spleen or blood tissue.
  - Evaluated classifier performance and selected the most effective models based on accuracy and computational efficiency.

# Repository Setup and Data Preparation

Due to the large size of the `.ipynb` files, it is recommended to clone the repository for optimal viewing and execution. Additionally, the data files located in the `data/` and `data/pooled_data` directories need to be unzipped to ensure the notebooks run without errors.

## Steps to Set Up the Repository

1. **Clone the Repository:**
   ```sh
   git clone <repository-url>

2. **Unzip data files**
   Navigate to the `data/` directory and unzip the `data.zip` file.
   Navigate to the `data/pooled_data/` directory and unzip the necessary files.

If you wish to run the `pooling_samples.ipynb` script specifically, it is sufficient to unzip only the `data.zip` file within the `data/` directory.

