import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utilities import train_regression_model, Model_dynamic, calculate_weighted_mse


def objective(trial, train_dataset, val_dataset, device, num_of_epochs):
    # Hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-7, 1e-3, log=True)  # Log scale search for learning rate
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])  # Discrete search for batch size
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)  # Log scale search for weight decay
    h1_values = trial.suggest_int('h1', 5, 13)
    num_layers = trial.suggest_int('num_layers', 1, 3)

    # Create DataLoader with the current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # Create model instance
    model = Model_dynamic(h1=h1_values, num_layers=num_layers)
    model.to(device)

    # Define the criterion and optimizer
    criterion = calculate_weighted_mse(reduction='mean')#nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    # Train the model
    trained_model, train_losses, val_losses, r2_mean = train_regression_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_of_epochs,
        device=device,
        patience=10,
        scheduler=scheduler
    )

    min_val_loss = min(val_losses)


    trial.set_user_attr("mean_r2", r2_mean)

    return min_val_loss  # Minimize the validation loss

def callback(study, trial, best_r2_trial):
    # Retrieve the `max_mean_r2` value from the user attributes
    max_mean_r2 = trial.user_attrs.get("max_mean_r2", -float('inf'))
    if max_mean_r2 > best_r2_trial['r2']:
        best_r2_trial['r2'] = max_mean_r2
        best_r2_trial['trial_number'] = trial.number
        best_r2_trial['params'] = trial.params
        print(f"New best R2 trial: {best_r2_trial['trial_number']} | R2: {best_r2_trial['r2']:.4f} | Params: {best_r2_trial['params']}")