import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from data_loader import MergedDataset
from model import Model, CombinedModel
from loss import WeightedMSE
from utils import train_regression_model


def train(gas, config_gas, config_arch, config_train, outputs_points, freeze_layers, dir_path, model_pth,
          device, geometry_layer, unfreeze_layers, layers_to_unfreeze, verbose=False):

    csv_file_path_train = f'./data/train_data_no_head_outer_corner_{gas}.csv'
    dataset = MergedDataset(csv_file_path_train, stats_file="stats_min_max.json", columns_idx=outputs_points)

    batch = config_gas['batch_size']
    lr = config_gas['lr']
    weight_decay = config_gas['weight_decay']

    neurons_per_layer = config_arch['neurons_per_layer']
    layers = config_arch['layers']

    epochs = config_train['epochs']
    patience = config_train['patience']

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Split the dataset not randomly, keep the same sets everytime
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    if geometry_layer:
        pretrained_models = []

        for model_idx in outputs_points:
            model_i = Model(h1=neurons_per_layer, num_layers=layers, freeze_layers=[], output_size=1)
            model_i.load_state_dict(torch.load(f'{dir_path}/trained_model_{model_idx}.pth'))
            pretrained_models.append(model_i)

        model = CombinedModel(h1=neurons_per_layer, num_layers=layers, pretrained_models=pretrained_models)
    else:
        model = Model(h1=neurons_per_layer,
                      num_layers=layers,
                      freeze_layers=freeze_layers,
                      output_size=len(outputs_points))

    if model_pth and not geometry_layer:
        model.load_state_dict(torch.load(model_pth))
        print("Loaded pre-trained model.")
        lr /= 20
        print("Adjusted learning rate.")

    criterion = WeightedMSE(reduction='mean', outputs_points=outputs_points, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    outputs_idx_str = '_'.join(map(str, outputs_points))

    start_time = time.time()
    trained_model, losses, val_losses, mean_r2_scores = train_regression_model(model=model,
                                                                               train_loader=train_loader,
                                                                               val_loader=val_loader,
                                                                               criterion=criterion,
                                                                               optimizer=optimizer,
                                                                               num_epochs=epochs,
                                                                               device=device,
                                                                               patience=patience,
                                                                               scheduler=scheduler,
                                                                               dir_path=dir_path,
                                                                               outputs_idx_str=outputs_idx_str,
                                                                               unfreeze_layers=unfreeze_layers,
                                                                               layers_to_unfreeze=layers_to_unfreeze,
                                                                               geometry_layer=geometry_layer)
    end_time = time.time()

    model_save_path = f'{dir_path}/trained_model_{outputs_idx_str}.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Time for training: {(end_time-start_time):.4f}")

    if verbose:
        plt.plot(list(range(len(losses))), losses, label="Training Loss")
        plt.plot(list(range(len(losses))), val_losses, label="Validation Loss")
        plt.ylabel("Total Loss")
        plt.xlabel("Epoch")
        plt.title("Training & Validation Loss Progression")
        plt.legend()
        plt.tick_params(axis='both', which='major')
        plt.show()

    return model_save_path, max(mean_r2_scores)
