import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def check_accuracy(loader, model,device,criterion=None):
    
    print("Checking model accuracy...")
    num_correct = 0
    num_samples = 0
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()
    if criterion:
        val_loss = criterion(scores, y)
        val_loss+=val_loss.item()*x.size(0)
        return val_loss



def load_model(path,device,num_classes):
    
    test_model = models.mobilenet_v3_small() # we do not specify pretrained=True, i.e. do not load default weights
    test_model.classifier = nn.Sequential(
    nn.Linear(test_model.classifier[0].in_features, 100), nn.Hardswish(),  nn.Linear(100, num_classes))
    
    test_model.load_state_dict(torch.load(path,map_location=torch.device(device)))
    
    test_model.to(device=device)
    test_model = test_model.eval()
    return test_model

