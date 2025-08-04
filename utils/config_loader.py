import yaml

def load_config(file_path):
    """
    Load configuration from a YAML file.
    
    Args:
        file_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration data as a dictionary.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config