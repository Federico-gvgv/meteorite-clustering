from sklearn.preprocessing import StandardScaler

def preprocess_meteorite_data(data):
    """
    Preprocess the meteorite dataset:
    - Encode categorical data.
    - Drop missing values.
    - Normalize numerical features.

    Args:
        data (pd.DataFrame): The raw dataset.

    Returns:
        pd.DataFrame, np.ndarray: Cleaned DataFrame and normalized features.
    """
    
    # Drop rows with missing essential values
    data = data.dropna(subset=['mass', 'reclat', 'reclong'])
    
    # Select relevant features
    features = data[['mass', 'reclat', 'reclong']]
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    return data, normalized_features
