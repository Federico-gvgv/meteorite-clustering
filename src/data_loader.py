import pandas as pd

def load_meteorite_data(csv_file_path):
    """
    Load the NASA Meteorite Landings dataset from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    data = pd.read_csv(csv_file_path)
    return data
