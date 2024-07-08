import torch

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
