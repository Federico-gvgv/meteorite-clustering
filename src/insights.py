def analyze_clusters(data, cluster_labels):
    """
    Analyze the mean feature values for each cluster.

    Args:
        data (pd.DataFrame): Cleaned dataset.
        cluster_labels (np.ndarray): Cluster labels for the data points.

    Returns:
        pd.DataFrame: Summary of mean values for each cluster.
    """
    # Add cluster labels to the dataset
    data['Cluster'] = cluster_labels

    # Select only numeric columns for the analysis
    numeric_data = data.select_dtypes(include=['number'])

    # Group by cluster and calculate mean values
    cluster_summary = numeric_data.groupby('Cluster').mean()
    return cluster_summary

