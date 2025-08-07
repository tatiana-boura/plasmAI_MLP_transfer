import os
import yaml
import pandas as pd

from scripts.utils import setup_device, set_seed
from scripts.train import train_GNN
from scripts.test import test_GNN

device = setup_device()

seed_num = 41
set_seed(seed_num)

# Open .yaml file to get configuration
with open('./config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

dir_path = f'./GNN'
os.makedirs(dir_path, exist_ok=True)

#graph_model='GATConv'
#graph_model='SAGEConv'
graph_model='GCNConv'

test_by = "power"
#test_by = "percentage"
df = pd.read_csv("./data/mixture_O2_Ar_dataset.csv", sep=';')

print(df.shape)

if test_by == "percentage":
    test_df = df[df['xAr'] < 0.3].copy()
    train_df = df[df['xAr'] >= 0.3].copy()
    print(train_df.shape, test_df.shape)
else:
    power_percentile = df["Power"].quantile(0.75)
    pressure_percentile = df["Pressure"].quantile(0.75)
    test_df = df[(df['Power'] >= power_percentile) & (df['Pressure'] >= pressure_percentile)].copy()
    train_df = df[(df['Power'] < power_percentile) | (df['Pressure'] < pressure_percentile)].copy()
    print(train_df.shape, test_df.shape)

print(f'\nTraining:\n')
train_GNN(train_df=train_df, config=config, dir_path=dir_path, device=device, test_by=test_by, graph_model=graph_model, verbose=False)
print(f'\nEvaluation:\n')
test_GNN(test_df=test_df, config=config, dir_path=dir_path, device=device, test_by=test_by, graph_model=graph_model, verbose=False)
