import config
import torch
from torch.utils.data import random_split,DataLoader,Subset
from torchvision.datasets import ImageFolder
from config import SEED,BATCH_SIZE

def get_subset(indices, start, end):
    return indices[start : start + end]

def split_data(folder_path = config.DATA_FOLDER,train_ratio=0.7):
    
    assert train_ratio <=0.8
    
    dataset_train = ImageFolder(folder_path, transform=config.train_transforms)
    dataset_val = ImageFolder(folder_path, transform=config.val_transforms)
    dataset_test = ImageFolder(folder_path, transform=config.val_transforms)

    TRAIN_PCT, VALIDATION_PCT = train_ratio, (1 - train_ratio)/2  # rest will go for test
    train_count = int(len(dataset_train) * TRAIN_PCT)
    validation_count = int(len(dataset_train) * VALIDATION_PCT)
    
    indices = torch.randperm(len(dataset_train),generator=torch.Generator().manual_seed(SEED))
    
    train_indices = get_subset(indices, 0, train_count)
    val_indices = get_subset(indices, train_count, validation_count)
    test_indices = get_subset(indices, train_count + validation_count, len(dataset_train))
    
    train_set = Subset(dataset_train, train_indices)
    val_set = Subset(dataset_val, val_indices)
    test_set = Subset(dataset_test, test_indices)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader,val_loader,test_loader