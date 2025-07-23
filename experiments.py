import os
import yaml
import pandas as pd

from scripts.utils import setup_device, set_seed
from scripts.train import train_single, train_mixture
from scripts.test import test_single, test_mixture

device = setup_device()
current_directory = os.path.dirname(os.path.realpath(__file__))

seed_num = 41
set_seed(seed_num)

# Open .yaml file to get configuration
with open('./config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


data_type = "mixture"

if data_type == "single":

    gas = 'O2'
    training_type = 'baseline'  # baseline, fine_tune, freeze
    stats_json = './data/stats_min_max.json'


    dir_path = f'./{gas}/{training_type}'
    os.makedirs(dir_path, exist_ok=True)

    print(f"Gas is {gas}.")

    print(f'\nTraining:\n')
    train_single(gas=gas, config=config, dir_path=dir_path, device=device, stats_json=stats_json, training_type=training_type, verbose=False)
    print(f'\nEvaluation:\n')
    test_single(gas=gas, config=config, dir_path=dir_path, device=device, stats_json=stats_json, verbose=False)

elif data_type == "mixture":

    gas = 'mixture'


    dir_path = f'./{gas}'
    os.makedirs(dir_path, exist_ok=True)


    # Load CSV
    df = pd.read_csv("./data/mixture_O2_Ar_dataset.csv", sep=';')

    # Split the testing data according to precentage xAr
    test_df = df[df['xAr'] > 0.8].copy()
    train_df = df[df['xAr'] <= 0.8].copy()

    print(f"Mixture of gases.")

    '''print(f'\nTraining:\n')
                train_mixture(train_df=train_df, gas=gas, config=config, dir_path=dir_path, device=device, verbose=False)'''
    print(f'\nEvaluation:\n')
    test_mixture(test_df=test_df, gas=gas, config=config, dir_path=dir_path, device=device, verbose=False)
    

else:
    raise ValueError("Unknown data type.")