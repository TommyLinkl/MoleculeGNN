import torch
import torch.nn as nn
import torch.optim as optim
from src.models_gnn import SchNet, MEGNet
from src.qm9_process import QM9Dataset
from src.train_eval import train, evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 32
    lr = 0.001
    num_epochs = 10

    dataset = QM9Dataset(root='./data/QM9', batch_size=batch_size)
    dataset.process_dataset(target=4)
    # dataset.print_instance()
    train_loader, val_loader, test_loader = dataset.get_dataloaders()
    
    data = next(iter(train_loader))
    model = SchNet(data)  # Initialize SchNet with a single instance of data
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Evaluate on test set
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
