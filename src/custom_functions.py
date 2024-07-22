import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from kneed import KneeLocator # locate the elbow of PCA variance plot
from sklearn.model_selection import ParameterGrid
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, \
                            recall_score, precision_score, matthews_corrcoef, roc_auc_score, \
                            f1_score, fbeta_score


# The functions `evaluate_tsne` and `evaluate_umap` are similar and function in the exact same manner.
# They loop through the parameter grid of the dimentionality reduction method (for t-SNE: perplexity and learning rate
# and for UMAP: number of neighbors and minimum distance). 
# A metric that does not demand clustering in order to evaluate an embedding is trustworthiness.
# Due to this fact, the function calculates the trustworthiness for each parameter combination and plots all the embeddings.
# The final evaluation is user-dependent and it is based on the visualization. 
# Also, a dictionary for trustworthiness is returned that has as keys the parameter combination just for inspection puproses.

def evaluate_tsne(adata, max_perplexity_number=50, learning_step=400, random_state=42, color=None):

    npcs = len(adata.uns['pca']['variance_ratio'])

    tsne_param_grid = {
    'perplexity': [5] + list(range(10, max_perplexity_number+1, 5)),
    'learning_rate': list(range(200, 1001, learning_step))
    }

    results = {}

    grid = ParameterGrid(tsne_param_grid)
    _ , axes = plot_creating(len(tsne_param_grid['perplexity'])*len(tsne_param_grid['learning_rate']))
    axes = axes.ravel()

    # Initialize counter that will be used for plotting
    k = -1
    
    for param_combo in tqdm(grid, desc="Evaluating t-SNE parameters", total=len(grid)):
        perplexity = param_combo['perplexity'] 
        learning_rate = param_combo['learning_rate']

        k += 1 # counter in order for each combination to correspond to an axes[k] for plotting
        sc.tl.tsne(adata, n_pcs=npcs, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
        trust = trustworthiness(adata.X, adata.obsm['X_tsne'])
        results[(perplexity, learning_rate)] = trust
        sc.pl.tsne(adata, ax=axes[k], color=color, title=f"Perp: {perplexity}, LR: {learning_rate}, Trust: {round(trust, 4)}", show=False)
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    return results

# with tqdm

def evaluate_umap(adata, max_neighbors=50, max_min_distance=1, random_state=42, color=None):
    umap_param_grid = {
        'neighbors': [5] + list(range(10, max_neighbors+1, 5)),
        'min_dist': [0.01] + list(np.arange(0.1, float(max_min_distance), 0.2))
    }

    grid = ParameterGrid(umap_param_grid)
    results = {}

    # Initialize the number of plots generated
    plot_num = len(umap_param_grid['neighbors']) * len(umap_param_grid['min_dist'])

    # The function plot creating is used
    _, axes = plot_creating(plot_num)
    axes = axes.ravel()

    # Initialization of k variable that will be used for plot organization 
    k = -1

    for param_combo in tqdm(grid, desc="Evaluating UMAP parameters", total=len(grid)):
        n_neighbors = param_combo['neighbors']
        min_dist = param_combo['min_dist']

        k += 1
        # Try different neighbor values
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=random_state)
        # Different minimum distances
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
        trust = trustworthiness(adata.X, adata.obsm['X_umap'])
        results[(n_neighbors, min_dist)] = trust
        # Visualize
        sc.pl.umap(adata, ax=axes[k], color=color, show=False, title=f"Neighbors={n_neighbors}, Min dist={min_dist}, Trust: {round(trust, 4)}")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    return results

# This function finds the parameters of the dimentionality reduction techniques of UMAP and t-SNE
# and also calculates the mean and variance of the trustworthiness metric across all the different embeddings

def best_hyperparams(dictionary, technique):

    max_value = max(dictionary.values())
    best_params = None
    mean_trust = np.mean(list(dictionary.values()))
    var_trust = np.var(list(dictionary.values()))

    print(f"The mean trustworthiness for {technique.upper()} is {mean_trust} while its variance is {var_trust}")

    for params, value in dictionary.items():
        if value == max_value:
            best_params = params
            break

    if technique.upper() == 'TSNE':
        perp, lern = best_params
        print(f'The best parameters for t-SNE based on the highest trustworthiness are perplexity {perp} and learning rate {lern}')
        # return perp, lern
    elif technique.upper() == 'UMAP':
        neighbors, min_dist = best_params
        print(f'The best parameters for UMAP based on the highest trustworthiness are number of neighbors {neighbors} and minimum distance {min_dist}')
        # return neighbors, min_dist
    else:
        print("The technique you provided is not supported")



# This function is used to show the PCA components and the variance of the data they can capture.
# It is used for vizualisation purposes.

def evaluate_PCA(adata, max_n_comp):
    variance_explained = []
    number_of_components_list = range(10, max_n_comp + 1, 5)

    for i in number_of_components_list:
        sc.tl.pca(adata, svd_solver='arpack', n_comps=i)
        variance_explained.append(sum(adata.uns['pca']['variance_ratio'])*100)

    plt.figure(figsize=(16, 8))
    plt.axhline(y=90, color='red', linestyle='--')  # Add a horizontal line at 90%
    plt.axvline(x=50, color='black')  # Add a horizontal line at 50 pcs
    plt.plot(number_of_components_list, variance_explained)
    plt.title('The % of variance of the data explained by its respective number of components')
    plt.xlabel("Number of components used")
    plt.xticks(ticks=range(0, max_n_comp + 1, 5))
    plt.ylabel("Explained variance of the data (%)")
    plt.show()

    return number_of_components_list, variance_explained

# This function has the purpose of locating the ideal number of components for PCA 
# as it constitutes a computational way to define the elbow of the scree plot. 

def locate_elbow(adata, show_plot=False):

    sc.tl.pca(adata, n_comps=50)

    x = range(1, len(adata.uns['pca']['variance']) + 1)
    y = adata.uns['pca']['variance']

    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    
    if show_plot:
        plt.plot(x, y)
        plt.axvline(x=kn.knee, color='red', linestyle='--')
        plt.title('Elbow detection of the number of components vs cumulative explained variance')
        plt.xlabel('Number of components')
        plt.ylabel('Explained Variance by Principal Component')
        plt.show()

    sc.tl.pca(adata, n_comps=kn.knee)

    return adata


# Create mupliple plots

def plot_creating(num):
    if num == 3:
        return plt.subplots(1, 3, figsize=(15, 5))
    elif num == 4:
        return plt.subplots(1, 4, figsize=(20, 5))
    elif num % 3 == 0:
        return plt.subplots(num//3, 3, figsize=(20, 5 * (num // 3)))
    elif num % 4 == 0:
        return plt.subplots(num//4, 4, figsize=(20, 5 * (num // 4)))
    elif num % 5 == 0:
        return plt.subplots(num//5, 5, figsize=(20, 5 * (num // 5)))
    elif num % 2 == 0:
        return plt.subplots(num//2, 2, figsize=(20, 5 * (num // 2)))
    else:
        return plt.subplots(num, 1, figsize=(6, num**2))

# A function so that the cells can be visualized with their respective labels

def plot_clusters_2d(embedding, labels, title="2D Cluster Visualization"):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    palette = ['g','r','b','m','y','c', 'orange','purple','pink','brown','lime','olive']
    for k, col in zip(unique_labels, palette):
        class_member_mask = (labels == k)
        xy = embedding[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'Cluster {k}', edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

# This function generates a dictionary. The key of the dictionary is the dimentionality reduction technique (in our case PCA, t-SNE and UMAP)
# while the respective value is the optimal embedding selected by the user based on the visualizations of evaluate_tsne and evaluate_umap.
# The option to plot the best embeddings is also given.
# Default values for t-SNE: perplexity=10 and learning rate=800.
# Default parameters for UMAP: number of neighbors=30 and minimum distance=0.01.
# The embedding of PCA is defined by the number of principal components that is calculated by the `locate_elbow` function.

def best_embeddings(adata, embedding_pcs, tsne_perplexity=10, tsne_learning=800, umap_neighbors=30, umap_min_dist=0.01, random_state=42, plot=False):

    embeddings = {}
    
    sc.pp.neighbors(adata, n_neighbors=umap_neighbors, random_state=random_state)
    sc.tl.umap(adata, min_dist=umap_min_dist, random_state=random_state)
    sc.tl.tsne(adata, perplexity=tsne_perplexity, learning_rate=tsne_learning, random_state=random_state)

    embeddings = {'PCA': adata.obsm['X_pca'][:, :embedding_pcs], 'UMAP': adata.obsm['X_umap'], 't-SNE': adata.obsm['X_tsne']}

    if plot:
        sc.pl.pca(adata, title='Best PCA embedding')
        sc.pl.umap(adata, title='Best UMAP embedding')
        sc.pl.tsne(adata, title='Best t-SNE embedding')
        
    return embeddings

# The GMMClustering class is designed to perform clustering on high-dimensional data using Gaussian Mixture Models (GMM). 
# It explores a grid of hyperparameters to identify the best model based on the Bayesian Information Criterion (BIC)
# and evaluates clustering quality using the Silhouette score.

class GMMClustering:
    def __init__(self, param_grid, random_state=42):
        self.param_grid = param_grid
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_bic = np.inf
        self.results = []

    def fit(self, embedding):
        embedding = StandardScaler().fit_transform(embedding)
        for params in ParameterGrid(self.param_grid):
            try:
                gmm = GaussianMixture(**params, random_state=self.random_state)
                gmm.fit(embedding)
                bic = gmm.bic(embedding)
                if bic < self.best_bic:
                    self.best_bic = bic
                    self.best_model = gmm
                    self.best_params = params
                self.results.append({
                    'params': params,
                    'bic': bic,
                    'silhouette': silhouette_score(embedding, gmm.predict(embedding))
                })
            except ValueError as e:
                print(f"Skipping parameters {params} due to error: {e}")

    def predict(self, embedding):
        if self.best_model is not None:
            embedding = StandardScaler().fit_transform(embedding)
            return self.best_model.predict(embedding)
        else:
            raise Exception("Model has not been fitted yet")

    def predict_proba(self, embedding):
        if self.best_model is not None:
            embedding = StandardScaler().fit_transform(embedding)
            return self.best_model.predict_proba(embedding)
        else:
            raise Exception("Model has not been fitted yet")

    def get_best_params(self):
        return self.best_params

    def get_results(self):
        return self.results
    

# A function to use the above class returning the probabilites and the labels of the winning algorithm dicated by bic

def GMM_fit(gmm_param_grid, myembedding):

    # Create and fit the GMMClustering class
    gmm_clustering = GMMClustering(gmm_param_grid)
    gmm_clustering.fit(myembedding)

    # Get the best model parameters and clustering results
    best_params = gmm_clustering.get_best_params()
    print(f"Best GMM parameters: {best_params}")

    clustering_results = gmm_clustering.get_results()
    clustering_df = pd.DataFrame(clustering_results)

    # Predict cluster labels and posterior probabilities
    labels = gmm_clustering.predict(myembedding)
    probas = gmm_clustering.predict_proba(myembedding)

    return labels, probas, clustering_df

# Same class as with GMM but the clustering algorithm is K-means. The evaluation metric in this case is also silhuette score.

class KMeansClustering:
    def __init__(self, param_grid, random_state=42):
        self.param_grid = param_grid
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_silhouette = -1  # Initialize to the lowest possible silhouette score
        self.results = []

    def fit(self, embedding):
        embedding = StandardScaler().fit_transform(embedding)
        for params in ParameterGrid(self.param_grid):
            try:
                kmeans = KMeans(**params, random_state=self.random_state)
                kmeans.fit(embedding)
                silhouette = silhouette_score(embedding, kmeans.labels_)
                if silhouette > self.best_silhouette:
                    self.best_silhouette = silhouette
                    self.best_model = kmeans
                    self.best_params = params
                self.results.append({
                    'params': params,
                    'silhouette': silhouette
                })
            except ValueError as e:
                print(f"Skipping parameters {params} due to error: {e}")

    def predict(self, embedding):
        if self.best_model is not None:
            embedding = StandardScaler().fit_transform(embedding)
            return self.best_model.predict(embedding)
        else:
            raise Exception("Model has not been fitted yet")

    def get_best_params(self):
        return self.best_params

    def get_results(self):
        return self.results
    
# Function to use the KMeansClustering class

def kmeans_fit(kmeans_param_grid, embedding):

    kmeans_clustering = KMeansClustering(kmeans_param_grid)
    kmeans_clustering.fit(embedding)

    best_params = kmeans_clustering.get_best_params()
    results = kmeans_clustering.get_results()

    # Print silhouette score for KMeans clustering (assuming labels_pca is defined)
    labels= kmeans_clustering.predict(embedding)

    return labels, best_params


# This function takes as input the cells and the labels exports a .csv file
# where the cells are the indexes with their respective label in the 'label' column

def make_csv_labels(adata, labels, output_name):

    # Cells are the index
    index = pd.Index(adata.obs.index)

    # Create a DataFrame with labels_of_best_model
    df = pd.DataFrame({'labels': labels}, index=index)

    if not output_name.endswith('.csv'):
        output_name += '.csv'

    df.to_csv("../labels/" + output_name)
    return df

def compute_metrics_old(y_true, y_pred, label_encoder=None):
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # metrics chosen
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # color palette for heatmap
    cmap = sns.color_palette("Blues")
    
    # get class labels from label_encoder if provided
    if label_encoder is not None:
        class_labels = label_encoder.classes_
    else:
        # default labels if label_encoder is not provided
        class_labels = ['Class 0', 'Class 1']

    # Plot confusion matrix using Seaborn heatmap
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Plot barplot of metrics
    plt.subplot(1, 2, 2)
    metrics = ['Accuracy', 'Balanced\nAccuracy', 'Recall', 'Precision', 'MCC', 'ROC AUC', 'F1', 'F2']
    values = [accuracy, balanced_acc, recall, precision, mcc, roc_auc, f1]
    bars = plt.bar(metrics, values, color=['blue', 'peru', 'green', 'purple', 'red', 'orange', 'pink'])
    plt.xticks(rotation=45)
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.2) 

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.2f}', ha='center', va='bottom')
    
    # Show plot
    plt.tight_layout()
    plt.show()

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'recall': recall,
        'precision': precision,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'f1': f1
    }

def compute_metrics(y_true, y_pred, label_encoder=None):
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # metrics chosen
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # color palette for heatmap
    cmap = sns.color_palette("Blues")
    
    # get class labels from label_encoder if provided
    if label_encoder is not None:
        class_labels = label_encoder.classes_
    else:
        # default labels if label_encoder is not provided
        class_labels = ['Class 0', 'Class 1']

    # Plot confusion matrix using Seaborn heatmap
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Plot barplot of metrics
    plt.subplot(1, 2, 2)
    metrics = ['Accuracy', 'Balanced\nAccuracy', 'Recall', 'Precision', 'MCC', 'ROC AUC', 'F1']
    values = [accuracy, balanced_acc, recall, precision, mcc, roc_auc, f1]
    bars = plt.bar(metrics, values, color=['blue', 'peru', 'green', 'purple', 'red', 'orange', 'pink'])
    plt.xticks(rotation=45)
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.2) 

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.2f}', ha='center', va='bottom')
    
    # Show plot
    plt.tight_layout()
    plt.show()

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'recall': recall,
        'precision': precision,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'f1': f1
    }
