from packages_classifier.model_nn import NeuralNet
from packages_classifier.train_nn import train_model
from packages_classifier.trained_nn import test_model
from packages_gio import prepare_data
import torch

def __dhelp__():
    import sys
    print("USAGE")
    print(f"\t{sys.argv[0]} [RUN ARGUMENTS] [PATH ARGUMENTS] [-t|n|k]")
    
    print("\nRUN ARGUMENTS")
    print(f"{' ': >4}{'-c': <22}{' ': >4}Attempt to use CUDA, training only")
    print(f"{' ': >30}an NVidia GPU must be available")
    print(f"{' ': >4}{'-u': <22}{' ': >4}Don't split loaded data into subsets")
    print(f"{' ': >30}Only for testing with trained model")
    print(f"{' ': >4}{'-r [ROWS_TO_READ]': <22}{' ': >4}How many rows to read from dataset")
    print(f"{' ': <32}Integer, defaults to None for all")
    print(f"{' ': >4}{'-s [ROWS_TO_SKIP]': <22}{' ': >4}How many rows to skip from dataset")
    print(f"{' ': >30}Integer, defaults to None for none")

    print("\nPATHS")
    print(f"{' ': >4}{'-f [DATASET_PATH]': <22}{' ': >4}Path to dataset, assumed csv format:")
    print(f"{' ': >30}[3072 bits of encoded data];[integer label:0-3]")
    print(f"{' ': >4}{'-m [MODEL_OUTPUT_PATH]': <22}{' ': >4}Path to directory for model output")
    print(f"{' ': >30}Useful only for TRAINING operation (-t flag)")
    print(f"{' ': >4}{'-i [SAVED_MODEL_PATH]': <22}{' ': >4}Path to pretrained model")
    print(f"{' ': >30}Useful only for RUN TRAINED operation (-n flag)")

    print("\nACTIONS (select one)")
    print(f"{' ': >4}{'-t': <22}{' ': >4}Train model")
    print(f"{' ': >4}{'-n': <22}{' ': >4}Run specified trained model")
    print(f"{' ': >30}requires -i to specify model path")
    print(f"{' ': >4}{'-k': <22}{' ': >4}Test untrained network")


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

def __train__(try_CUDA=False, dataset_path="./dataset/dataset.csv", models_path="./dataset/dataset.csv", nrows=None, skiprows=None):
    model = NeuralNet()
    # model.set_verbose(2)

    if try_CUDA and model.gpu_available():
        print("Attempting to load NeuralNet to CUDA...")
        model.empty_gpu_cache()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        model = model.to(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    
    print(f"NeuralNet params on CUDA: {next(model.parameters()).is_cuda}")

    print("Reading data...")
    X_train, y_train, X_val, y_val, _, _ = prepare_data(dataset_path, nrows, skiprows, split_dataset = True)
    print("Data read successfully")
    
    # TRAIN MODEL
    train_model(model, X_train, y_train, X_val, y_val, models_path=models_path)
    # SAVE MODEL
    # model.save_network("model_fi.pt")
    pass

def __trained__(try_CUDA=False, dataset_path="./dataset/dataset.csv", trained_model_path="./model_cp_76.pt", nrows=None, skiprows=None, split_dataset = True):
    model = NeuralNet()
    model.load_network(trained_model_path)

    if try_CUDA and model.gpu_available():
        print("Attempting to load NeuralNet to CUDA...")
        model.empty_gpu_cache()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # torch.backends.cuda.matmul.allow_tf32 = True

    print(f"NeuralNet params on CUDA: {next(model.parameters()).is_cuda}")
    
    print("Reading data...")
    _, _, _, _, X_test, y_test = prepare_data(dataset_path, nrows, skiprows, split_dataset)
    print("Data read successfully")

    test_model(model, X_test, y_test)
    pass

def __run__():
    import sys
    n = len(sys.argv)
    
    if n < 2:
        __dhelp__()
        exit()

    try_CUDA = False
    dataset_path = "./dataset/dataset.csv"
    models_path = "./models/"
    trained_model_path = "./models/model_cp_76.pt"
    nrows = None
    skiprows = None
    actions = [0,0,0]
    split_trained = True

    i = 1

    while i < n:
        match sys.argv[i]:
            case '-c':
                try_CUDA = True
            case '-u':
                split_trained = False
            case '-f':
                dataset_path = sys.argv[i+1]
                i +=1
                pass
            case '-m':
                models_path = sys.argv[i+1]
                i +=1
            case '-i':
                trained_model_path = sys.argv[i+1]
                i += 1
                pass
            case '-r':
                nrows = int(sys.argv[i+1])
                i +=1
            case '-s':
                skiprows = int(sys.argv[i+1])
                i +=1
            case '-t':
                actions[0] += 1
                pass
            case '-n':
                actions[1] += 1
                pass
            case '-k':
                actions[2] += 1
            case _:
                __dhelp__()
                print(f"Unrecognized flag: {sys.argv[i]}")
                raise ValueError("ERROR: Unknown flag")
        i += 1
        pass
    if sum(actions) != 1:
        __dhelp__()
        raise ValueError("ERROR: Select exactly one action argument")
    match actions:
        case [1,0,0]:
            __train__(try_CUDA, dataset_path, models_path, nrows, skiprows)
            pass
        case [0,1,0]:
            __trained__(try_CUDA, dataset_path, trained_model_path, nrows, skiprows, split_trained)
            pass
        case [0,0,1]:
            __test__()
            pass
    pass


if __name__ == "__main__":
    __run__()