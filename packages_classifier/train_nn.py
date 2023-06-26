import torch
from torch import nn, optim

def train_model(model, X_train, y_train, X_val, y_val, epochs=700, batch_size=10):
    # Convert lists to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    # Create data loaders
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs_u = inputs.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs_u)
            outputs_s = torch.squeeze(outputs,1)

            loss = criterion(outputs_s, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs_u = inputs.unsqueeze(1)

                val_loss += criterion(torch.squeeze(model(inputs_u)), labels)
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item()} - Val loss: {val_loss.item()}')
