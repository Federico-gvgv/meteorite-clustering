from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_clusters_pca(normalized_features, cluster_labels, output_path=None):
    """
    Visualize clusters using PCA for dimensionality reduction.

    Args:
        normalized_features (np.ndarray): Normalized dataset.
        cluster_labels (np.ndarray): Cluster labels.
        output_path (str): Path to save the visualization.
    Returns:
        None
    """
    # Reduce dimensions to 2D
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(normalized_features)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=cluster_labels, palette='tab10')
    plt.title("Meteorite Landings Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")

    # Save the plot or show it
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def visualize_clusters_tsne(normalized_features, cluster_labels, output_path=None):
    """
    Visualize clusters using t-SNE for dimensionality reduction.

    Args:
        normalized_features (np.ndarray): Normalized dataset.
        cluster_labels (np.ndarray): Cluster labels.
        output_path (str): Path to save the visualization.

    Returns:
        None
    """
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    reduced_data = tsne.fit_transform(normalized_features)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=cluster_labels, palette='tab10')
    plt.title("Meteorite Landings Clustering (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster")
    
    # Save the plot or show it
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
