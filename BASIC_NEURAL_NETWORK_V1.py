import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
print("packages imported successfully")

#Create a model class that inherits nn.Module
class Model(nn.Module):
    #input layer ( 2 features, Power Pressure) --> Hidden Layer 1 (H1) --> H2 --> Output (25 outputs)
    def __init__(self, in_features=2, h1=10, h2=10, out_features=25):
        super().__init__() # instantiate our nn.Module, always have to do it
        self.fc1 = nn.Linear(in_features, h1) #we suppose fully connected layer (fc-> fully connected)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    #we need now the function to move everything forward
    def forward(self, x):
        #we choose relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Pick a manual seed for randomization
torch.manual_seed(41)

#create an instance for the model
basic_model = Model()

# load our data
df = pd.read_csv('merged_1-2116.csv', sep=';')

# statistics of data
for column in df.columns:
    col_data = df[column]  # save the column data, df.columns shows the column header
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

#now normalize the data
df_norm = df.copy()
for column in df_norm.columns:
    df_norm[column] = (df_norm[column] - df_norm[column].min()) / ( df_norm[column].max() - df_norm[column].min())
X_columns = ['Power', 'Pressure']  # features
y_columns = [col for col in df_norm.columns if col not in X_columns]  # To be predicted
X, y = df_norm[X_columns], df_norm[y_columns]
#convert to numpy arrays
X = X.values
y = y.values

#split the data for test and train
from sklearn.model_selection import train_test_split
seed_init = 41
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed_init)
# use 15% of data for testing and 85% for training
#now we need to convert everything to tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

#MEASURE ERROR
criterion = nn.CrossEntropyLoss()
#CHOOSE ADAM OPTIMIZER
optimizer = torch.optim.Adam(basic_model.parameters(), lr=0.0001)

# Train the model
# Epoch -> (one run thru all the training data in our network)
epochs = 50000
losses = []  # empty array, we will append everything
for i in range(epochs):
    y_pred = basic_model.forward(X_train)  # Get predicted results

    # measure the loss
    loss = criterion(y_pred, y_train)  # predicted values vs the y_train
    losses.append(loss.detach().numpy())
    # print every 10 epoch
    if i % 10000 == 0:
        print(f'Epoch num: {i} loss: {loss}')

    optimizer.zero_grad()  # set the gradients to zero
    loss.backward()
    optimizer.step()


plt.plot(range(epochs), losses)
plt.ylabel("error")
plt.xlabel("Epoch")
