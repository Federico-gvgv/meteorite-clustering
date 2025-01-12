from src.data_loader import load_meteorite_data
from src.preprocessing import preprocess_meteorite_data
from src.clustering import perform_kmeans, perform_elbow_analysis, compute_silhouette_score
from src.visualization import visualize_clusters_pca, visualize_clusters_tsne
from src.insights import analyze_clusters

# Load dataset
data = load_meteorite_data('data/meteorite_landings.csv')

# Preprocess the data
data, normalized_features = preprocess_meteorite_data(data)

# Perform elbow analysis
perform_elbow_analysis(normalized_features)

# Perform clustering
cluster_labels = perform_kmeans(normalized_features, n_clusters=4)

# Compute silhouette score
compute_silhouette_score(normalized_features, cluster_labels)

# Visualize clusters
visualize_clusters_pca(normalized_features, cluster_labels)

# Visualize clusters with t-SNE
visualize_clusters_tsne(normalized_features, cluster_labels, output_path='results/meteorite_clusters_tsne.png')

# Analyze and save insights
cluster_summary = analyze_clusters(data, cluster_labels)
print(cluster_summary)
cluster_summary.to_csv('results/meteorite_cluster_summary.csv', index=False)