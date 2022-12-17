import torch.nn as nn
import timm
class CNN(nn.Module):
    def __init__(self,N,magistral):
        super(CNN, self).__init__()
        self.model = timm.create_model(magistral, pretrained=True)
        self.classification = nn.Linear(1000,N)
        self.relu = nn.ReLU()
    def forward(self,x):
        x =                    self.relu(self.model(x))
        classification =       self.classification(x)
        return classification