import yaml
import os

from utils import set_seed, setup_device
from train import train
from test import test

device = setup_device()

seed_num = 41
set_seed(seed_num)

# Open .yaml file to get configuration
with open('./config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

gas = 'Ar'
training_type = 'baseline'  # baseline, fine_tune, freeze

dir_path = f'./{gas}/{training_type}'
os.makedirs(dir_path, exist_ok=True)

config_Ar = config[gas]
config_arch = config['nn_arch']
config_train = config['training']

outputs_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

saved_model_path = None

for i, outputs_point in enumerate(outputs_points):

    print(f'Training output point {outputs_point}')

    freeze_layers = [] if i == 0 else [0, 1, 2]

    saved_model_path = train(gas=gas, outputs_points=[outputs_point], freeze_layers=freeze_layers,
                             model_pth=saved_model_path, config_gas=config_Ar, config_arch=config_arch,
                             config_train=config_train, device=device, dir_path=dir_path)


'''test(gas=gas, config_arch=config_arch, outputs_points=outputs_points, freeze_layers=freeze_layers,
     dir_path=dir_path, trained_pth=saved_model_path, device=device)'''
