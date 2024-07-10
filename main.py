import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from src.models_gnn import SchNet, MEGNet
from src.qm9_process import QM9Dataset
from src.train_eval import train, evaluate, plot_training_validation_cost

def main():
    # NVIDIA , CUDA support / MPS (Apple Silicon GPU) support
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS is available. \n")
        device = torch.device('mps')
    elif torch.cuda.is_available():
        print("CUDA GPU is available. \n")
        device = torch.device('cuda')
    else:
        print("Only CPU is available. This will be slow. \n")
        device = torch.device('cpu')
    device = torch.device('cpu')
    
    # Reading calculation information + hyperparameters
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    print(f"Done reading in the config.yml file. The job type is {config['Job']['run_mode']}. The model is {config['Model']['model']}. \n")
    num_epochs = config['Model']['num_epochs']

    os.makedirs(config['Job']['output_directory'], exist_ok = True)

    dataset = QM9Dataset(root='./data/QM9', batch_size=config['Model']['batch_size'])
    dataset.process_dataset(target=4)
    # dataset.print_instance()
    train_loader, val_loader, test_loader = dataset.get_dataloaders()
    
    data = next(iter(train_loader))
    MODEL_CLASSES = {
        'SchNet': SchNet,
        'MEGNet': MEGNet,
    }
    model_class = MODEL_CLASSES.get(config['Model']['model'])
    if model_class:
        model = model_class(data)   # Initialize SchNet with a single instance of data, since the construction of the GNN models require a data instance
        model = model.to(device)
    else:
        raise NotImplementedError(f"Model {config['Model']['model']} not implemented.")
    optimizer = getattr(optim, config['Model']['optimizer'])(model.parameters(), lr=config['Model']['lr'], **config['Model']["optimizer_args"])
    scheduler = getattr(optim.lr_scheduler, config['Model']['scheduler'])(optimizer, **config['Model']['scheduler_args'])
    criterion = nn.MSELoss()

    # Training loop
    training_COST = []
    validation_COST = []
    best_val_error = float('inf')
    best_epoch = 0
    file_trainCost = open(f"{config['Job']['output_directory']}training_cost.dat", 'w')
    file_valCost = open(f"{config['Job']['output_directory']}validation_cost.dat", 'w')
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        end_time = time.time()
        scheduler.step(val_loss)

        training_COST.append(train_loss)
        file_trainCost.write(f"{epoch + 1}  {train_loss}\n")
        file_trainCost.flush()
        validation_COST.append(val_loss)
        file_valCost.write(f"{epoch + 1}  {val_loss}\n")
        file_valCost.flush()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}. Runtime: {end_time - start_time:.3f}s")

        # Save model checkpoint if validation loss improves
        if val_loss < best_val_error:
            best_val_error = val_loss
            best_epoch = epoch
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'full_model': model,
            }, f"{config['Job']['output_directory']}best_model.pth")

        # Early stopping logic
        if epoch - best_epoch > config['Model']['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch + 1} as validation loss has not improved for {config['Model']['early_stopping_patience']} epochs.")
            break

    file_trainCost.close()
    file_valCost.close()

    # Plotting training and validation costs
    fig_cost = plot_training_validation_cost(training_COST, validation_COST, False)
    fig_cost.savefig(f"{config['Job']['output_directory']}train_val_cost.png")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
