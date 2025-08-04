import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
import time
import os
import matplotlib.pyplot as plt

from scripts.data_loader import MergedDataset, MixtureDataset
from scripts.model import Model, MixtureEtchModel, MixtureGNN
from scripts.loss import WeightedMSE
from scripts.utils import train_regression_model


def train_GNN(train_df, config, dir_path, device, test_by, graph_model, verbose=False):

    dataset = MixtureDataset(train_df, set_type="train", gnn=True)

    single_neurons_per_layer = config['single_nn_arch']['neurons_per_layer']
    single_layers = config['single_nn_arch']['layers']

    batch = config['GNN']['batch_size'] 
    lr = config['GNN']['lr'] 
    weight_decay = config['GNN']['weight_decay'] 

    epochs = config['training']['epochs'] 
    patience = config['training']['patience'] 

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Split the dataset not randomly, keep the same sets everytime
    '''train_dataset = Subset(dataset, range(0, train_size))
                val_dataset = Subset(dataset, range(train_size, len(dataset)))'''

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # Load the pre-trained sinlge element networks
    model_Ar = Model(h1=single_neurons_per_layer, num_layers=single_layers, input_size=2)
    model_O2 = Model(h1=single_neurons_per_layer, num_layers=single_layers, input_size=2)

    model_Ar.load_state_dict(torch.load('./Ar/baseline/trained_model.pth', weights_only=False))
    model_O2.load_state_dict(torch.load('./O2/baseline/trained_model.pth', weights_only=False))

    # Make GNN model
    model_gnn = MixtureGNN(graph_model=graph_model)

    # Make wrapper mixture GNN model
    full_model = MixtureEtchModel(fnn_a=model_Ar, fnn_b=model_O2, gnn=model_gnn, device=device)

    total = sum(p.numel() for p in full_model.parameters())
    trainable = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")

    criterion = WeightedMSE(reduction='mean', device=device)

    optimizer = torch.optim.Adam(full_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    start_time = time.time()
    trained_model, losses, val_losses, r2_mean = train_regression_model(model=full_model,
                                                                        train_loader=train_loader,
                                                                        val_loader=val_loader,
                                                                        criterion=criterion,
                                                                        optimizer=optimizer,
                                                                        num_epochs=epochs,
                                                                        device=device,
                                                                        patience=patience,
                                                                        scheduler=scheduler,
                                                                        dir_path=dir_path)
    end_time = time.time()
    model_save_path = f'{dir_path}/trained_model_{test_by}.pth'
    model_save_path_gnn = f'{dir_path}/trained_model_gnn_{test_by}.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    torch.save(trained_model.gnn.state_dict(), model_save_path_gnn)
    
    print(f"model saved to {model_save_path}")
    print(f"time for training: {(end_time-start_time):.4f}")

    plt.plot(list(range(len(losses))), losses, label="Training Loss")
    plt.plot(list(range(len(losses))), val_losses, label="Validation Loss")
    plt.ylabel("Total Loss")
    plt.xlabel("Epoch")
    plt.title("Training & Validation Loss Progression")
    plt.legend()
    plt.tick_params(axis='both', which='major')
    plt.savefig(f"{dir_path}/loss_progression_{test_by}.png", dpi=600, bbox_inches='tight')
    if verbose:
        plt.show()


def train_mixture(train_df, gas, config, dir_path, device, test_by, verbose=False):

    dataset = MixtureDataset(train_df, set_type="train")

    neurons_per_layer = config['mixture_nn_arch']['neurons_per_layer']
    layers = config['mixture_nn_arch']['layers']
    batch = config[gas]['batch_size'] 
    lr = config[gas]['lr'] 
    weight_decay = config[gas]['weight_decay'] 
    epochs = config['training']['epochs'] 
    patience = config['training']['patience'] 

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Split the dataset not randomly, keep the same sets everytime
    '''train_dataset = Subset(dataset, range(0, train_size))
                val_dataset = Subset(dataset, range(train_size, len(dataset)))'''

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    model = Model(h1=neurons_per_layer, num_layers=layers, input_size=3)

    criterion = WeightedMSE(reduction='mean', device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    start_time = time.time()
    trained_model, losses, val_losses, r2_mean = train_regression_model(model=model,
                                                                        train_loader=train_loader,
                                                                        val_loader=val_loader,
                                                                        criterion=criterion,
                                                                        optimizer=optimizer,
                                                                        num_epochs=epochs,
                                                                        device=device,
                                                                        patience=patience,
                                                                        scheduler=scheduler,
                                                                        dir_path=dir_path)
    end_time = time.time()
    model_save_path = f'{dir_path}/trained_model_{test_by}.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"model saved to {model_save_path}")
    print(f"time for training: {(end_time-start_time):.4f}")

    plt.plot(list(range(len(losses))), losses, label="Training Loss")
    plt.plot(list(range(len(losses))), val_losses, label="Validation Loss")
    plt.ylabel("Total Loss")
    plt.xlabel("Epoch")
    plt.title("Training & Validation Loss Progression")
    plt.legend()
    plt.tick_params(axis='both', which='major')
    plt.savefig(f"{dir_path}/loss_progression_{test_by}.png", dpi=600, bbox_inches='tight')
    if verbose:
        plt.show()



def train_single(gas, config, dir_path, device, stats_json, training_type, verbose=False):


    csv_file_path_train = f'./data/train_data_no_head_outer_corner_{gas}.csv'
    dataset = MergedDataset(csv_file_path_train, statsfile=stats_json)


    neurons_per_layer = config['single_nn_arch']['neurons_per_layer']
    layers = config['single_nn_arch']['layers']
    batch = config[gas]['batch_size'] 
    lr = config[gas]['lr'] 
    weight_decay = config[gas]['weight_decay'] 
    epochs = config['training']['epochs'] 
    patience = config['training']['patience'] 

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Split the dataset not randomly, keep the same sets everytime
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)

    model = Model(h1=neurons_per_layer, num_layers=layers)

    if training_type == "fine_tuning":
        neural_network_pth = "./pass"
        model.load_state_dict(torch.load(neural_network_pth))

        lr /= 20

        print("Loaded pre-trained model and adjusted lr")

    criterion = WeightedMSE(reduction='mean', device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    start_time = time.time()
    trained_model, losses, val_losses, r2_mean = train_regression_model(model=model,
                                                                        train_loader=train_loader,
                                                                        val_loader=val_loader,
                                                                        criterion=criterion,
                                                                        optimizer=optimizer,
                                                                        num_epochs=epochs,
                                                                        device=device,
                                                                        patience=patience,
                                                                        scheduler=scheduler,
                                                                        dir_path=dir_path)
    end_time = time.time()
    model_save_path = f'{dir_path}/trained_model.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"model saved to {model_save_path}")
    print(f"time for training: {(end_time-start_time):.4f}")

    plt.plot(list(range(len(losses))), losses, label="Training Loss")
    plt.plot(list(range(len(losses))), val_losses, label="Validation Loss")
    plt.ylabel("Total Loss")
    plt.xlabel("Epoch")
    plt.title("Training & Validation Loss Progression")
    plt.legend()
    plt.tick_params(axis='both', which='major')
    plt.savefig(f"{dir_path}/loss_progression.png", dpi=600, bbox_inches='tight')
    if verbose:
        plt.show()
