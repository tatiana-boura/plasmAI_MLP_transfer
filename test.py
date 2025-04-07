import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_loader import MergedDataset, unscale_min_max
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

from loss import WeightedMSE
from utils import test_model
from model import Model, CombinedModel


def test(gas, config_arch, outputs_points, freeze_layers, dir_path, trained_pth, geometry_layer, device):

    csv_file_path_test = f'data/test_data_no_head_outer_corner_{gas}.csv'

    neurons_per_layer = config_arch['neurons_per_layer']
    layers = config_arch['layers']

    stats_json = 'stats_min_max.json'

    dataset_test = MergedDataset(csv_file=csv_file_path_test, stats_file=stats_json, columns_idx=outputs_points)
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    if geometry_layer:
        pretrained_models = []

        for model_idx in outputs_points:
            model_i = Model(h1=neurons_per_layer, num_layers=layers, freeze_layers=[], output_size=1)
            model_i.load_state_dict(torch.load(f'{dir_path}/trained_model_{model_idx}.pth'))
            pretrained_models.append(model_i)

        trained_model = CombinedModel(h1=neurons_per_layer, num_layers=layers, pretrained_models=pretrained_models)
        trained_model.load_state_dict(torch.load(trained_pth))

    else:
        trained_model = Model(h1=neurons_per_layer, num_layers=layers,
                              freeze_layers=freeze_layers, output_size=len(outputs_points))
        trained_model.load_state_dict(torch.load(trained_pth))


    criterion = WeightedMSE(reduction='mean', outputs_points=outputs_points, device=device)

    test_loss, all_predictions, all_targets, r22, _ = test_model(model=trained_model,
                                                                 test_loader=test_loader,
                                                                 criterion=criterion,
                                                                 device=device)
    print(f"Mean test loss: {test_loss:.4f}")

    with open(stats_json, 'r') as f:
        scaling_info = json.load(f)

    output_columns_names = list(scaling_info.keys())
    output_columns_names = [col for col in output_columns_names if col not in ['Power', 'Pressure']]

    outputs_idx_str = '_'.join(map(str, outputs_points))

    unscaled_predictions = unscale_min_max(all_predictions, output_columns_names, scaling_info)
    predictions_df = pd.DataFrame(unscaled_predictions, columns=output_columns_names)
    predictions_df_csv = f'{dir_path}/pred_{outputs_idx_str}.csv'
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
    plt.savefig(f"{dir_path}/r2_{outputs_idx_str}.png", dpi=600, bbox_inches='tight')
    plt.close()

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
    plt.savefig(f"{dir_path}/residuals_{outputs_idx_str}.png", dpi=600, bbox_inches='tight')
    plt.close()
