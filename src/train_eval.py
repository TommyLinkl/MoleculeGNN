import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.num_graphs
    return running_loss / len(train_loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y.view(-1))
            running_loss += loss.item() * data.num_graphs
    return running_loss / len(loader.dataset)


def plot_training_validation_cost(training_cost, validation_cost, ylogBoolean, SHOWPLOTS=False): 
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    
    epochs = range(0, len(training_cost))
    axs.plot(np.array(epochs)+1, training_cost, "b-", label='Training Cost')

    if len(validation_cost) != 0: 
        evaluation_frequency = len(training_cost) // len(validation_cost)
        evaluation_epochs = list(range(evaluation_frequency-1, len(training_cost), evaluation_frequency))
        axs.plot(np.array(evaluation_epochs)+1, validation_cost, "r:", label='Validation Cost')

    if ylogBoolean:
        axs.set_yscale('log')
    else:
        axs.set_yscale('linear')
    axs.set(xlabel="Epochs", ylabel="Cost", title="Training and Validation Costs")
    axs.legend(frameon=False)
    axs.grid(True)
    fig.tight_layout()
    if SHOWPLOTS:
        plt.show()
    return fig


def inference(model, data_loader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            # Assuming data.id and data.value are accessible attributes
            for i in range(len(data.id)):
                result = {
                    'id': data.id[i].item(),  # Convert to Python scalar
                    'value': data.value[i].item(),  # Convert to Python scalar
                    'output': output[i].item()  # Convert to Python scalar
                }
                results.append(result)

    # Convert results list to DataFrame
    df = pd.DataFrame(results)
    return df