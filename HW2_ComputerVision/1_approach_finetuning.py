import torch
import torch.nn as nn
import torch.optim as optim
from dataset_setup import prepare_data_and_model
from trainer import train_model, evaluate_model, plot_confusion_matrix, plot_correct_and_incorrect_predictions
import time
import os

train_loader, val_loader, test_loader, train_subset, val_subset, model, device = prepare_data_and_model()

parametri_addestrabili = 0

for parametro in model.parameters():
    if parametro.requires_grad == True:
        parametri_addestrabili += parametro.numel()
print(f"Parametri addestrabili: {parametri_addestrabili}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4) # Learning rate 0.0001

print("\nInizio addestramento")
start_time = time.time()

model = train_model(model, train_loader, val_loader, train_subset, val_subset, criterion, optimizer, num_epochs=5, device=device)

end_time = time.time()
print(f"Tempo di addestramento: {end_time - start_time:.2f} secondi")

all_labels, all_preds = evaluate_model(model, test_loader, device)

nomi_classi = train_subset.dataset.classes
plot_confusion_matrix(all_labels, all_preds, nomi_classi, titolo_grafico="Approach 1 (Full Fine-Tuning)")

plot_correct_and_incorrect_predictions(model, test_loader, nomi_classi, device)


# Salvo checkpoint
base_dir = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(base_dir, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True) 
percorso_file = os.path.join(checkpoints_dir, 'checkpoint_approach_1.pth')
torch.save(model.state_dict(), percorso_file)
print(f"Checkpoint salvato correttamente in: {percorso_file}")
