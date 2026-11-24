import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import numpy as np
from sklearn.model_selection import train_test_split
import copy 
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Windows / Linux
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
    # MAC
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def prepare_data_and_model():
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Oxford-IIIT Pet dataset
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms() 

    train_dataset = datasets.OxfordIIITPet(root="./data", split="trainval", download=True, transform=transform)
    test_dataset = datasets.OxfordIIITPet(root="./data", split="test", download=True, transform=transform)

    model = models.mobilenet_v3_small(weights=weights).to(device)

    # Suddivisione del dataset
    targets = train_dataset._labels # classe 
    indices = np.arange(len(train_dataset)) # indici immagini

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets, # Bilanciamento delle classi
        random_state=42
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 

    print("Dim train: ", len(train_loader)) 
    print("Dim val: ", len(val_loader))
    print("Dim test: ", len(test_loader))   # 115 batch, ogniuno composto da 32 immagini 

    print("Modello classifier: ", model.classifier)
    # print("Modello features: ", model.features)

    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 37).to(device) # nn.Linear(in_features, out_features)

    print("Modello nuovo: ", model.classifier) # Ora ho 37 classi

    print("\nConteggio immagini")
    train_counts = np.bincount([train_dataset._labels[i] for i in train_indices])
    val_counts = np.bincount([train_dataset._labels[i] for i in val_indices])
    print(f"Immagini per classe in Train: {min(train_counts)} - {max(train_counts)}")
    print(f"Immagini per classe in Validation: {min(val_counts)} - {max(val_counts)}")
    print(f"Totale Test set: {len(test_dataset)} immagini")

    return train_loader, val_loader, test_loader, train_subset, val_subset, model, device


