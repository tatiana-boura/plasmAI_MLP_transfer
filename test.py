import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_loader import MergedDatasetTest, unscale_min_max
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

from loss import WeightedMSE
from utils import setup_device, test_model
from model import Model
import os

device = setup_device()

gas = 'Ar'
training_type = 'baseline'  # baseline, fine_tune, freeze

dir_path = f'./{gas}/{training_type}'
os.makedirs(dir_path, exist_ok=True)

current_directory = os.path.dirname(os.path.realpath(__file__))
csv_file_path_test = f'data/test_data_no_head_outer_corner_{gas}.csv'
trained_pth = f'{dir_path}/trained_model.pth'

h1_val = 10
layers = 3

stats_json = 'stats_min_max.json'

dataset_test = MergedDatasetTest(csv_file_path_test, stats_json)
test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

trained_model = Model(h1=h1_val, num_layers=layers, freeze_layers=[])
trained_model.load_state_dict(torch.load(trained_pth))

criterion = WeightedMSE(reduction='mean', device=device)

test_loss, all_predictions, all_targets, r22, _ = test_model(model=trained_model,
                                                             test_loader=test_loader,
                                                             criterion=criterion,
                                                             device=device)
print(f"Mean test loss: {test_loss:.4f}")

with open(stats_json, 'r') as f:
    scaling_info = json.load(f)

output_columns_names = list(scaling_info.keys())
output_columns_names = [col for col in output_columns_names if col not in ['Power', 'Pressure']]

unscaled_predictions = unscale_min_max(all_predictions, output_columns_names, scaling_info)
predictions_df = pd.DataFrame(unscaled_predictions, columns=output_columns_names)
predictions_df_csv = f'{dir_path}/pred.csv'
predictions_df.to_csv(predictions_df_csv, index=False, sep=';')

model_predictions = pd.read_csv(predictions_df_csv, sep=';')
real_targets = pd.read_csv(csv_file_path_test, usecols=lambda column: column not in ['Power', 'Pressure'], sep=';')

r2_scores = []
for col in range(model_predictions.shape[1]):
    pred_col = model_predictions.iloc[:, col]
    target_col = real_targets.iloc[:, col]

    r2 = r2_score(target_col, pred_col)
    r2_scores.append(r2)

plt.figure(figsize=(10.5, 7.5))
plt.bar(list(range(1, len(r2_scores) + 1)), r2_scores, color='skyblue', edgecolor='black')
plt.xlabel('Etching rate point')
plt.ylabel(r"$R^2$ Score")
plt.title(r"$R^2$ Scores")
plt.xticks(list(range(1, len(r2_scores) + 1)))
plt.tight_layout()
plt.tick_params(axis='both', which='major')
plt.savefig(f"{dir_path}/r2.png", dpi=600, bbox_inches='tight')
plt.show()

# Visualize distribution of residuals
residuals = abs(real_targets - model_predictions)/real_targets * 100
residuals_flatten = residuals.values.flatten()
plt.figure(figsize=(10.5, 7.5))
ax = sns.kdeplot(residuals, color='blue', fill=False)

# Remove "O" from legend labels
new_labels = [label.get_text().replace("O", "") for label in ax.get_legend().get_texts()]
ax.legend(new_labels)
plt.title('Histogram of Residuals')
plt.xlabel('(physics_based – NN)/physics_based %')
plt.ylabel('Frequency')
plt.xlim(-1, 100)    # Set the x-axis to extend to 15%
plt.tick_params(axis='both', which='major')
plt.savefig(f"{dir_path}/residuals.png", dpi=600, bbox_inches='tight')
plt.show()
