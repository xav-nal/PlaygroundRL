import yaml
from easydict import EasyDict

def cfg_from_yaml_file(cfg_file):
    print("config", cfg_file)
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    return EasyDict(new_config)


