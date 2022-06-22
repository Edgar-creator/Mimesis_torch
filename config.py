import torch
from torchvision import  transforms
import torchvision.transforms.functional as TF
import numpy as np

SEED = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
DATA_FOLDER = 'rotation_test'

mean=[0.485, 0.456, 0.406]#0, 0, 0]#
std=[0.229, 0.224, 0.225]#[1,1,1]#

class SquarePad:
    
    def __call__(self, image):
        
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        
        return TF.pad(image, padding, 0, 'constant')


train_transforms = transforms.Compose(
    [
     SquarePad(),
     transforms.RandomRotation(degrees=10),
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(
         mean=mean,
         std=std)
        ]
        )

val_transforms = transforms.Compose(
    [
     SquarePad(),
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=mean,
        std=std)
        ]
        )
