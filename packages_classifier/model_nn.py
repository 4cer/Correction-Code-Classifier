import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Pre-inception first convs
        self.s1_conv_1x3 = nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,padding=1)
        self.s2_conv_1x3 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.s3_maxp_1x3 = nn.MaxPool1d(kernel_size=3)

        # Inception
        # Rail 1
        self.r1_conv_1x1 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        self.r1v2_conv_1x1 = nn.Conv1d(in_channels=256,out_channels=64,kernel_size=1,padding=0)

        # Rail 2
        self.r2_conv_1x1 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        self.r2v2_conv_1x1 = nn.Conv1d(in_channels=256,out_channels=64,kernel_size=1,padding=0)
        self.r2_conv_1x3 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,padding=1)

        # Rail 3
        self.r3_conv_1x1 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        self.r3v2_conv_1x1 = nn.Conv1d(in_channels=256,out_channels=64,kernel_size=1,padding=0)
        self.r3_conv_1x5 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=5,padding=2)

        # Rail 4
        self.r4_maxp_1x2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.r4_conv_1x1 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        self.r4v2_conv_1x1 = nn.Conv1d(in_channels=256,out_channels=64,kernel_size=1,padding=0)

        # Rail 5
        self.r5_maxp_1x2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.r5_conv_1x3 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.r5v2_conv_1x3 = nn.Conv1d(in_channels=256,out_channels=64,kernel_size=3,padding=1)

        # Inception pooling
        self.rx_maxp_1x2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Post-inception
        self.p1_avgp_16x1 = nn.AvgPool1d(kernel_size=16)
        self.p2_dropout = nn.Dropout(p=0.3)
        self.p3_linear = nn.Linear(in_features=256, out_features=128)
        self.p4_linear = nn.Linear(in_features=128, out_features=4)
        

    def forward(self, x):
        print(f"{'DIM INPUT:': <30} {np.shape(x).__str__(): >20}")

        # Fusion - CNN piece
        print("PRE INCEPTIONS")
        x = F.relu(self.s1_conv_1x3(x))
        print(f"{'DIM after CONV 1x3-p1:': <30} {np.shape(x).__str__(): >20}")
        x = F.relu(self.s2_conv_1x3(x))
        print(f"{'DIM after CONV 1x3-p1:': <30} {np.shape(x).__str__(): >20}")
        x = self.s3_maxp_1x3(x)
        print(f"{'DIM after MAXP 1x3-p0:': <30} {np.shape(x).__str__(): >20}")

        # Fusion inception(s)
        print("INCEPTION 1")
        x = self.__inception0(x)
        print("INCEPTION 2")
        x = self.__inception(x)
        print("INCEPTION 3")
        x = self.__inception(x)

        print("POST INCEPTIONS")
        x = self.p1_avgp_16x1(x)
        print(f"{'DIM after AVGP 16x1-p0:': <30} {np.shape(x).__str__(): >20}")
        x = self.p2_dropout(x)
        print(f"{'DIM after DROPOUT:': <30} {np.shape(x).__str__(): >20}")

        # Reshape (batch_size,256,1) to (batch_size,1,256)
        x = x.view(x.size(0), 1, -1)
        print(f"{'DIM after VIEW 256,128:': <30} {np.shape(x).__str__(): >20}")

        x = self.p3_linear(x)
        print(f"{'DIM after LIN 1:': <30} {np.shape(x).__str__(): >20}")
        x = F.softmax(self.p4_linear(x), dim=2)
        print(f"{'DIM after S(LIN 2):': <30} {np.shape(x).__str__(): >20}")

        return x
    
    def __inception0(self, x):
        print(f"\t{'DIM INPUT:': <30} {np.shape(x).__str__(): >20}")
        
        r1 = x;r2 = x;r3 = x;r4 = x;r5 = x
        # Rail 1
        print("\tRAIL 1")
        r1 = F.relu(self.r1_conv_1x1(r1))
        print(f"\t{'DIM after CONV 1x1-p0:': <30} {np.shape(r1).__str__(): >20}")

        # Rail 2
        print("\tRAIL 2")
        r2 = F.relu(self.r2_conv_1x1(r2))
        print(f"\t{'DIM after CONV 1x1-p0:': <30} {np.shape(r2).__str__(): >20}")
        r2 = F.relu(self.r2_conv_1x3(r2))
        print(f"\t{'DIM after CONV 1x3-p1:': <30} {np.shape(r2).__str__(): >20}")

        # Rail 3
        print("\tRAIL 3")
        r3 = F.relu(self.r3_conv_1x1(r3))
        print(f"\t{'DIM after CONV 1x1-p0:': <30} {np.shape(r3).__str__(): >20}")
        r3 = F.relu(self.r3_conv_1x5(r3))
        print(f"\t{'DIM after CONV 1x5-p2:': <30} {np.shape(r3).__str__(): >20}")
        
        # Rail 4
        print("\tRAIL 4")
        r4 = F.relu(self.r4_maxp_1x2(r4))
        print(f"\t{'DIM after MAXP 1x2-p0:': <30} {np.shape(r4).__str__(): >20}")
        r4 = F.relu(self.r4_conv_1x1(r4))
        print(f"\t{'DIM after CONV 1x1-p0:': <30} {np.shape(r4).__str__(): >20}")
        
        # Rail 5
        print("\tRAIL 5")
        r5 = F.relu(self.r5_maxp_1x2(r5))
        print(f"\t{'DIM after MAXP 1x2-p0:': <30} {np.shape(r5).__str__(): >20}")
        r5 = F.relu(self.r5_conv_1x3(r5))
        print(f"\t{'DIM after CONV 1x3-p1:': <30} {np.shape(r5).__str__(): >20}")

        # Rail 45
        print("\tRAIL 4 CAT 5")
        r45 = torch.cat((r4, r5), dim=2)
        print(f"\t{'DIM after CAT:': <30} {np.shape(r45).__str__(): >20}")

        # Concatenate and return
        print("\tCONCATENATE")
        concatenated = torch.cat((r1, r2, r3, r45), dim=1)
        print(f"\t{'DIM after CAT:': <30} {np.shape(concatenated).__str__(): >20}")
        concatenated = self.rx_maxp_1x2(concatenated)
        print(f"\t{'DIM after MAXP 1x2-p0:': <30} {np.shape(concatenated).__str__(): >20}")

        return concatenated
    
    def __inception(self, x):
        print(f"\t{'DIM INPUT:': <30} {np.shape(x).__str__(): >20}")
        
        r1 = x;r2 = x;r3 = x;r4 = x;r5 = x
        # Rail 1
        print("\tRAIL 1")
        r1 = F.relu(self.r1v2_conv_1x1(r1))
        print(f"\t{'DIM after CONV 1x1-p0:': <30} {np.shape(r1).__str__(): >20}")

        # Rail 2
        print("\tRAIL 2")
        r2 = F.relu(self.r2v2_conv_1x1(r2))
        print(f"\t{'DIM after CONV 1x1-p0:': <30} {np.shape(r2).__str__(): >20}")
        r2 = F.relu(self.r2_conv_1x3(r2))
        print(f"\t{'DIM after CONV 1x3-p1:': <30} {np.shape(r2).__str__(): >20}")

        # Rail 3
        print("\tRAIL 3")
        r3 = F.relu(self.r3v2_conv_1x1(r3))
        print(f"\t{'DIM after CONV 1x1-p0:': <30} {np.shape(r3).__str__(): >20}")
        r3 = F.relu(self.r3_conv_1x5(r3))
        print(f"\t{'DIM after CONV 1x5-p2:': <30} {np.shape(r3).__str__(): >20}")
        
        # Rail 4
        print("\tRAIL 4")
        r4 = F.relu(self.r4_maxp_1x2(r4))
        print(f"\t{'DIM after MAXP 1x2-p0:': <30} {np.shape(r4).__str__(): >20}")
        r4 = F.relu(self.r4v2_conv_1x1(r4))
        print(f"\t{'DIM after CONV 1x1-p0:': <30} {np.shape(r4).__str__(): >20}")
        
        # Rail 5
        print("\tRAIL 5")
        r5 = F.relu(self.r5_maxp_1x2(r5))
        print(f"\t{'DIM after MAXP 1x2-p0:': <30} {np.shape(r5).__str__(): >20}")
        r5 = F.relu(self.r5v2_conv_1x3(r5))
        print(f"\t{'DIM after CONV 1x3-p1:': <30} {np.shape(r5).__str__(): >20}")

        # Rail 45
        print("\tRAIL 4 CAT 5")
        r45 = torch.cat((r4, r5), dim=2)
        print(f"\t{'DIM after CAT:': <30} {np.shape(r45).__str__(): >20}")

        # Concatenate and return
        print("\tCONCATENATE")
        concatenated = torch.cat((r1, r2, r3, r45), dim=1)
        print(f"\t{'DIM after CAT:': <30} {np.shape(concatenated).__str__(): >20}")
        concatenated = self.rx_maxp_1x2(concatenated)
        print(f"\t{'DIM after MAXP 1x2-p0:': <30} {np.shape(concatenated).__str__(): >20}")

        return concatenated



def test_model(model, sample_input):
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
        # print(probs[0])
        # print(probs[1])


def __test__():
    sample_input = np.zeros((10,384),dtype=np.float16)
    model = NeuralNet()

    # TRAIN MODEL
    

    # # SAVE MODEL
    # torch.save(model.state_dict(), "model.pt")

    # # LOAD MODEL
    # model.load_state_dict(torch.load("model.pt"))
    # model.eval()

    test_model(model, sample_input)


if __name__ == "__main__":
    __test__()
