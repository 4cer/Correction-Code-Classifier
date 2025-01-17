from packages_classifier.model_nn import NeuralNet
from packages_classifier.train_nn import train_model
from packages_classifier.trained_nn import test_model
from packages_gio import prepare_data


def __dhelp__():
    print("COMMANDS")
    print(f"{'-t': >6}{' ': >6}Train model")
    print(f"{'-n': >6}{' ': >6}Run trained model")
    print(f"{' ': >12}only possible if model.pt exists")
    print(f"{'-k': >6}{' ': >6}Test untrained network")


def __test__():
    import numpy as np
    sample_input = np.zeros((10,3072),dtype=np.float16)
    model = NeuralNet()
    model.set_verbose(2)
    
    import torch
    with torch.no_grad():
        # Convert sample input to a tensor
        sample_input = torch.tensor(sample_input).unsqueeze(1).float()
        
        # Pass the input through the model and get the predicted logits
        logits = model(sample_input)
        
        # Apply softmax to the logits to get class probabilities
        probs = logits.squeeze(0)
        
        # Print the predicted probabilities for each class
        for prob in probs:
            print(prob)

def __train__():
    # X_train, y_train, X_val, y_val, _, _ = prepare_data("./packages_datasetgen/demofile.txt")
    X_train, y_train, X_val, y_val, _, _ = prepare_data("./dataset/datasetBIG.csv")

    import numpy as np

    model = NeuralNet()
    # model.set_verbose(2)

    if model.gpu_available():
        model.empty_gpu_cache()
        dev = model.get_device()
        model = model.to(dev)
    
    print(f"NeuralNet params on CUDA: {next(model.parameters()).is_cuda}")
    
    # TRAIN MODEL
    train_model(model, X_train, y_train, X_val, y_val)
    # SAVE MODEL
    # model.save_network("model_fi.pt")
    pass

def __trained__():
    # _, _, _, _, X_test, y_test = prepare_data("./packages_datasetgen/demofile.txt")
    _, _, _, _, X_test, y_test = prepare_data("./dataset/dataset.csv")
    # _, _, _, _, X_test, y_test = prepare_data("./dataset/datasetBIG.csv")
    model = NeuralNet()
    model.load_network("model_cp_76.pt")

    test_model(model, X_test, y_test)
    pass

def __run__():
    import sys
    n = len(sys.argv)
    
    if n < 2:
        __dhelp__()
        exit()

    for i in range(1,n):
        match sys.argv[i]:
            case '-t':
                __train__()
                pass
            case '-n':
                __trained__()
                pass
            case '-k':
                __test__()
            case _:
                __dhelp__()
    pass


if __name__ == "__main__":
    __run__()