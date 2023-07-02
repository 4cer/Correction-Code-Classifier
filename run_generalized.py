from packages_classifier.model_nn import NeuralNet
from packages_classifier.train_nn import train_model
from packages_classifier.trained_nn import test_model
from packages_gio import prepare_data
import torch

def __dhelp__():
    import sys
    print("USAGE IN ORDER")
    print(f"\t{sys.argv[0]} [FLAGS] [-f DATASET_PATH] [-m MODEL_OUTPUT_PATH] [-i SAVED_MODEL_PATH] [-s ROWS_TO_SKIP] [-r ROWS_TO_READ] [-t|n|k]")
    
    print("\nRUN ARGUMENTS")
    print(f"{'-c': >6}{' ': >6}Attempt to use CUDA, training only")
    print(f"{' ': >12}an NVidia GPU must be available")
    print(f"{'-r [ROWS_TO_READ]': >6}{' ': >6}How many rows to read from dataset")
    print(f"{' ': >12}Integer, defaults to None for all")
    print(f"{'-s [ROWS_TO_SKIP]': >6}{' ': >6}How many rows to skip from dataset")
    print(f"{' ': >12}Integer, defaults to None for none")

    print("\nPATHS")
    print(f"{'-f [DATASET_PATH]': >6}{' ': >6}Path to dataset, assumed csv format:")
    print(f"{' ': >12}[3072 bits of encoded data];[integer label:0-3]")
    print(f"{'-m [MODEL_OUTPUT_PATH]': >6}{' ': >6}Path to directory for model output")
    print(f"{' ': >12}Useful only for TRAINING operation (-t flag)")
    print(f"{'-i [SAVED_MODEL_PATH]': >6}{' ': >6}Path to pretrained model")
    print(f"{' ': >12}Useful only for RUN TRAINED operation (-n flag)")

    print("\nACTIONS")
    print("select precisely one")
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
    X_train, y_train, X_val, y_val, _, _ = prepare_data(dataset_path, nrows, skiprows)
    print("Data read successfully")
    
    # TRAIN MODEL
    train_model(model, X_train, y_train, X_val, y_val, models_path=models_path)
    # SAVE MODEL
    # model.save_network("model_fi.pt")
    pass

def __trained__(dataset_path="./dataset/dataset.csv", trained_model_path="./model_cp_76.pt", nrows=None, skiprows=None):
    model = NeuralNet()
    model.load_network(trained_model_path)

    print("Reading data...")
    _, _, _, _, X_test, y_test = prepare_data(dataset_path, nrows, skiprows)
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
    trained_model_path = "./model_cp_76.pt"
    nrows = None
    skiprows = None

    i = 1

    while i < n:
        match sys.argv[i]:
            case '-c':
                try_CUDA = True
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
                __train__(try_CUDA, dataset_path, models_path, nrows, skiprows)
                pass
            case '-n':
                __trained__(dataset_path, trained_model_path, nrows, skiprows)
                pass
            case '-k':
                __test__()
            case _:
                __dhelp__()
        i += 1
    pass


if __name__ == "__main__":
    __run__()