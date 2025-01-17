import torch
from torch import nn, optim
from datetime import datetime
import gc

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=16, learning_rate=2e-3, models_path="./dataset/dataset.csv"):
    # Convert lists to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    gc.collect()

    # Load to GPU
    if next(model.parameters()).is_cuda:
        print("Transfering data to CUDA...")
        dev = model.get_device()
        X_train = X_train.to(dev)
        y_train = y_train.to(dev)
        X_val = X_val.to(dev)
        y_val = y_val.to(dev)
    print(f"Data is on CUDA: {X_train.is_cuda and y_train.is_cuda and X_val.is_cuda and y_val.is_cuda}")
    gc.collect()
    
    # Create data loaders
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(  optimizer,
    #                                         step_size=10,
    #                                         gamma=2e-4,
    #                                         last_epoch=-1,
    #                                         verbose=True)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=0.8,
        last_epoch=-1,
        verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    best_vloss = 1_000_000.
    
    # Training loop
    for epoch in range(epochs):
        model.train(True)
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs_u = inputs.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs_u)
            outputs_s = torch.squeeze(outputs,1)

            loss = criterion(outputs_s, labels)
            loss.backward()
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100
                print(f"\tbatch {i+1: <8} loss: {last_loss: >10.4f}")
                running_loss = 0

        # Validation
        model.eval()
        running_vloss = 0
        with torch.no_grad():
            for j, vdata in enumerate(val_loader):
                inputs, labels = vdata
                inputs_u = inputs.unsqueeze(1)

                running_vloss += criterion(torch.squeeze(model(inputs_u)), labels)
            
        avg_vloss = running_vloss / (j + 1)

        scheduler.step()
        
        # print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item()} - Val loss: {running_vloss.item()}')
        print(f'Epoch {epoch+1}/{epochs}: LOSS Training: {last_loss} - Validation: {avg_vloss} - Learning rate: {optimizer.param_groups[0]["lr"]}')

        # Track best performance and dump model
        if avg_vloss < best_vloss:
            print("Saving new best")
            best_vloss = avg_vloss
            torch.save(model.state_dict(), f"{models_path}{datetime.now().strftime('%m%d-%H%M')}_model_{avg_vloss:.8f}_cp.pt")
