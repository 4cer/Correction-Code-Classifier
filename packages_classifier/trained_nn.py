import torch
from torch import nn, optim

def tell_max(output, label):
    argmax = torch.argmax(output)
    return 1 if label.item() == argmax else 0


def print_confusion_matrix(confusion_matrix: torch.tensor):
    print("\nCONFUSION MATRIX\n")
    print(f"{'Prediction': >25}")
    print(f"{' ': <15}{0: <8}{1: <8}{2: <8}{3: <8}\n")
    for i, row in enumerate(confusion_matrix):
        if i == 0:
            print(f"{'Reference': <10}{i: <4}", end="")
        else:
            print(f"{' ': <10}{i: <4}", end="")
        for col in row:
            print(f"{col: < 8}", end="")
            pass
        print("")
        pass

    pass


def test_model(model, X_test, y_test):
    # Convert lists to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Load to GPU
    dev = model.get_device()
    if next(model.parameters()).is_cuda:
        print("Transfering data to CUDA...")
        X_test = X_test.to(dev)
        y_test = y_test.to(dev)
    print(f"Data is on CUDA: {X_test.is_cuda and y_test.is_cuda}")
    
    # Create data loaders
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True)
    
    # Define the optimizer and loss function
    criterion = nn.CrossEntropyLoss()

    confusion_matrix = torch.zeros([4,4], dtype=torch.long, device=dev)
    
    # Testing loop
    model.eval()
    with torch.no_grad():
        loss = 0.
        accuracy = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs_u = inputs.unsqueeze(1)

            outputs = model(inputs_u)
            outputs_s = torch.squeeze(outputs,1)

            loss += criterion(outputs_s, labels)
            accuracy += tell_max(outputs_s, labels)
            confusion_matrix[labels.item()][torch.argmax(outputs_s)] = confusion_matrix[labels.item()][torch.argmax(outputs_s)] + 1
            pass

    print(f"Average loss: {loss / (i+1)}")
    print(f"Accuracy: {accuracy / (i+1)}")
    print_confusion_matrix(confusion_matrix)
