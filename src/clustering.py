from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def perform_kmeans(normalized_features, n_clusters=3):
    """
    Apply K-Means clustering to the meteorite dataset.

    Args:
        normalized_features (np.ndarray): Normalized dataset.
        n_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Cluster labels for each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_features)
    return cluster_labels

def perform_elbow_analysis(normalized_features, max_clusters=10, output_path=None):
    """
    Perform the elbow analysis to determine the optimal number of clusters.

    Args:
        normalized_features (np.ndarray): The normalized dataset.
        max_clusters (int): The maximum number of clusters to evaluate.
        output_path (str): Path to save the plot.

    Returns:
        None
    """
    inertia = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_features)
        inertia.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

def compute_silhouette_score(normalized_features, cluster_labels):
    """
    Compute the silhouette score for the clustering.

    Args:
        normalized_features (np.ndarray): The normalized dataset.
        cluster_labels (np.ndarray): Cluster labels assigned by the clustering algorithm.

    Returns:
        float: The silhouette score.
    """
    score = silhouette_score(normalized_features, cluster_labels)
    print(f"Silhouette Score: {score}")
    return score