import torch
import json
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

class MixtureDataset(Dataset):
    def __init__(self, df, set_type):
        
        input_cols = ['Power', 'Pressure', 'xAr']
        X = df[input_cols].to_numpy()
        Y = df[[col for col in df.columns if col not in input_cols]].to_numpy()

        scaler_pth = "./data/scaler_mixture.pkl"

        if set_type == "train":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            joblib.dump(scaler, scaler_pth)
        else:
            scaler = joblib.load(scaler_pth)
            X = scaler.transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]

    def __len__(self):
        return self.X.shape[0]


class MergedDataset(Dataset):
    def __init__(self, csv_file, statsfile):
        df = pd.read_csv(csv_file, sep=';')
        # Read the min-max statistics from JSON
        with open(statsfile, 'r') as f:
            stats = json.load(f)

        # Normalize the data with min-max using stored statistics
        for column in df.columns:
            if column in stats:
                df[column] = (df[column] - stats[column]['min']) / (stats[column]['max'] - stats[column]['min'])

        X_columns = ['Power', 'Pressure']  # features
        y_columns = [col for col in df.columns if col not in X_columns]  # To be predicted
        self.X, self.y = df[X_columns], df[y_columns]
        self.df = df

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]


class MergedDatasetTest(Dataset):
    def __init__(self, csv_file, stats_file):
        df = pd.read_csv(csv_file, sep=';')
        # Load the pre-calculated mean and std statistics from the JSON file
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        for column in df.columns:
            if column in stats:
                df[column] = (df[column] - stats[column]['min']) / (stats[column]['max'] - stats[column]['min'])

        X_columns = ['Power', 'Pressure']  # features
        y_columns = [col for col in df.columns if col not in X_columns]  # To be predicted
        self.X, self.y = df[X_columns], df[y_columns]

        self.df = df

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]


def unscale_min_max(data, column_names, scaling_info):
    unscaled_data = data
    num_columns = data.shape[1]
    for i, column in enumerate(column_names):
        if i >= num_columns:  # Check if index exceeds the number of columns
            print(f"Warning: Index {i} is out of bounds for data with {num_columns} columns.")
            break
        min_ = scaling_info[column]['min']
        max_ = scaling_info[column]['max']
        unscaled_data[:, i] = (data[:, i] * (max_-min_)) + min_  # Reverse normalization
    return unscaled_data