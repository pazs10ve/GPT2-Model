import yaml

def read_config_yaml(file_path):

    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
