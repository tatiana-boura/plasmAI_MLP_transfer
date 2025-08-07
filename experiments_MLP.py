import os
import yaml
import pandas as pd

from scripts.utils import setup_device, set_seed
from scripts.train import train_single, train_mixture
from scripts.test import test_single, test_mixture

device = setup_device()

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
    test_by = 'power'
    #test_by = 'percentage'

    dir_path = f'./{gas}'
    os.makedirs(dir_path, exist_ok=True)

    # Load CSV
    df = pd.read_csv("./data/mixture_O2_Ar_dataset.csv", sep=';')

    if test_by == "percentage":
        test_df = df[df['xAr'] < 0.3].copy()
        train_df = df[df['xAr'] >= 0.3].copy()
    else:
        power_percentile = df["Power"].quantile(0.75)
        pressure_percentile = df["Pressure"].quantile(0.75)
        test_df = df[(df['Power'] >= power_percentile) & (df['Pressure'] >= pressure_percentile)].copy()
        train_df = df[(df['Power'] < power_percentile) | (df['Pressure'] < pressure_percentile)].copy()

    print(f"Mixture of gases.")

    print(f'\nTraining:\n')
    train_mixture(train_df=train_df, gas=gas, config=config, dir_path=dir_path, device=device, test_by=test_by, verbose=False)
    print(f'\nEvaluation:\n')
    test_mixture(test_df=test_df, gas=gas, config=config, dir_path=dir_path, device=device, test_by=test_by, verbose=False)
    

else:
    raise ValueError("Unknown data type.")