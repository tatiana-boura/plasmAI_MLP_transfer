import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
print("successfully imported packages")

class MergedDataset(Dataset):
    def __init__(self):
        df = pd.read_csv('merged_1-2116.csv', sep=';')
        #normalize the data with min-max
        df_copy = df.copy()
        for column in df_copy.columns:
            df_copy[column] = (df_copy[column] - df_copy[column].min()) / ( df_copy[column].max() - df_copy[column].min())
        X_columns = ['Power', 'Pressure']  # features
        y_columns = [col for col in df_copy.columns if col not in X_columns]  # To be predicted
        self.X, self.y = df_copy[X_columns], df_copy[y_columns]

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def print_column_stats(df):
        for column in df.columns:
            col_data = df[column]  # save the column data, df.columns shows the column headers

            # statistics calculations
            nan_percentage = col_data.isna().mean() * 100  # True indicates the presence of null or missing values and False indicates otherwise, from pandas
            min_value = col_data.min()
            max_value = col_data.max()
            mean_value = col_data.mean()
            std_value = col_data.std()

            # now we print the above
            print(f"Column: {column}")
            print(f"nan_percentage: {nan_percentage:.2f}%")
            print(f"Max value: {max_value}")
            print(f"Min value: {min_value}")
            print(f"average value: {mean_value}")
            print(f"std: {std_value}")
            print("-" * 30)  # print a line -----------





#MergedDataset(Dataset).print_column_stats()

train_data = MergedDataset()
dataloader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
train_data.print_column_stats()

for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print("Data:", data)
    print("Labels:", labels)