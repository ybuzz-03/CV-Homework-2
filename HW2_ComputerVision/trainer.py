import torch
import copy
import os
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def train_model(model, train_loader, val_loader, train_subset, val_subset, criterion, optimizer, num_epochs, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoca {epoch + 1} / {num_epochs}")
        print("-" * 15)

        # ADDESTRAMENTO
        model.train()

        if not any(p.requires_grad for p in model.features.parameters()):
            model.features.eval() # Blocco aggiornamento train BatchNorm

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images) # 37 punteggi per ogni classe, associati alla singola immagine --> outputs[0] (1 foto) mi da 37 punteggi cioe per ogni classe.
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0) 
            _, predicted = torch.max(outputs, 1) # Prendo indice classe "vincente". Estraggo il valore piu alto di outputs.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_subset)
        train_acc = 100 * correct / total

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # VALIDATION
        model.eval() # Blocco il dropout
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():   # Disattivo calcolo gradienti
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_subset)
        epoch_val_acc = 100.0 * val_correct / val_total
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nMiglior accuratezza in Validazione: {best_val_acc:.2f}%")
    model.load_state_dict(best_model_wts) # Terminate epoche ricarica nella rete i pesi migliori salvati
    
    return model


def evaluate_model(model, test_loader, device): 
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Spostamento risultati da GPU alla CPU 
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, zero_division=0)
    print(report)

    return all_labels, all_preds


def plot_confusion_matrix(all_labels, all_preds, nomi_classi, titolo_grafico):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=nomi_classi, 
                yticklabels=nomi_classi,
                annot_kws={"size": 8}) 
    plt.title(f"Matrice di Confusione - {titolo_grafico}", fontsize=18, pad=20)
    plt.xlabel("Classe Predetta", fontsize=14, labelpad=10)
    plt.ylabel("Classe Reale", fontsize=14, labelpad=10)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()


    nome_file = titolo_grafico.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    percorso_immagine = os.path.join(plots_dir, nome_file)
    plt.savefig(percorso_immagine, bbox_inches='tight', dpi=300)
    print(f"Matrice di confusione salvata in: {percorso_immagine}")

    plt.show()



def plot_correct_and_incorrect_predictions(model, test_loader, class_names, device, num_images=4):
    model.eval()
    correct_images, correct_preds, correct_labels = [], [], []
    incorrect_images, incorrect_preds, incorrect_labels = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                img = images[i].cpu().numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)

                if predicted[i] == labels[i] and len(correct_images) < num_images:
                    correct_images.append(img)
                    correct_preds.append(class_names[predicted[i]])
                    correct_labels.append(class_names[labels[i]])
                elif predicted[i] != labels[i] and len(incorrect_images) < num_images:
                    incorrect_images.append(img)
                    incorrect_preds.append(class_names[predicted[i]])
                    incorrect_labels.append(class_names[labels[i]])
                
                if len(correct_images) == num_images and len(incorrect_images) == num_images:
                    break
            if len(correct_images) == num_images and len(incorrect_images) == num_images:
                break

    fig, axes = plt.subplots(2, num_images, figsize=(15, 7))
    fig.suptitle("Esempi di Predizioni: Corrette (Sopra) e Sbagliate (Sotto)", fontsize=16)

    for i in range(num_images):
        axes[0, i].imshow(correct_images[i])
        axes[0, i].set_title(f"Pred: {correct_preds[i]}\nReale: {correct_labels[i]}", color="green")
        axes[0, i].axis("off")
        
        # Disegna le sbagliate
        axes[1, i].imshow(incorrect_images[i])
        axes[1, i].set_title(f"Pred: {incorrect_preds[i]}\nReale: {incorrect_labels[i]}", color="red")
        axes[1, i].axis("off")

    plt.tight_layout()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    percorso_immagine = os.path.join(plots_dir, "Approach_1_Predictions.png")
    plt.savefig(percorso_immagine, bbox_inches='tight', dpi=300)
    print(f"Griglia predizioni salvata in: {percorso_immagine}")

    plt.show()