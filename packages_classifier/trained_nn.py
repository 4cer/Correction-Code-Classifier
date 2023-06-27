import torch
from torch import nn, optim

def tell_max(output, label):
    argmax = torch.argmax(output)
    return 1 if label.item() == argmax else 0
    
    


def test_model(model, X_test, y_test):
    # Convert lists to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Create data loaders
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True)
    
    # Define the optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    
    # Testing loop
    model.eval()
    with torch.no_grad():
        loss = 0.
        accuracy = 0
        all1 = 0
        last = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs_u = inputs.unsqueeze(1)

            outputs = model(inputs_u)
            outputs_s = torch.squeeze(outputs,1)

            loss += criterion(outputs_s, labels)
            accuracy += tell_max(outputs_s, labels)
            if torch.equal(last, outputs_s):
                all1 += 1
            last = outputs_s
            pass
    
    # tell_max(outputs_s, labels)

    print(f"Average loss: {loss / (i+1)}")
    print(f"Accuracy: {accuracy / (i+1)}")
    #print(f"All the same: {all1 / (i+1)}")
