Our analysis aims to align with the methods used by the original authors, but since they did not provide code or parameter references for t-SNE, we conducted a comprehensive inspection of parameter grids for both t-SNE and UMAP. After downloading all data, we replicated the analysis pipeline for mice samples across several notebooks:

pooling_samples.ipynb: Samples were pooled in various combinations (all together, all murine spleen, all murine blood) and exported as .h5ad files in the /data/pooled_data folder. 
This approach streamlines sample loading and eliminates the need for repeated pooling in subsequent notebooks.

all_mouse_samples.ipynb: Pooled mice samples were loaded and subjected to t-SNE and UMAP analysis. Tissue-specific marker gene detection was also performed for spleen and blood tissues.

spleen_individual_inspection.ipynb / blood_individual_inspection.ipynb: Each individual mouse spleen donor was analyzed separately to explore how different t-SNE and UMAP parameters affect embedding and visualization.

spleen_pooled_inspection.ipynb / blood_pooled_inspection.ipynb: Analysis focused on pooled spleen samples to determine optimal t-SNE and UMAP parameters for effective visualization. 
After selecting the best parameters according to our judgement we eployed clustering algorithms such as k-means, gmm and hierarchical clustering 
to see if the clusters found are the same number as the authors'. 
However, the authors' labels for the clusters were not provided so we cannot be sure whether or not this separation has the same biological meaning.
In an attempt to dig further, we also found the driving genes of each cluster in order to compare them with the respective genes of the authors.

machine_learning_pipeline.ipynb: Different classifiers are used in order to classify cells to spleen or blood tissue. 
