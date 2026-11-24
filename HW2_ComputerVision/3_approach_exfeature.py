import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
from dataset_setup import prepare_data_and_model
from trainer import plot_confusion_matrix
from sklearn.metrics import classification_report
import time
import os
import joblib

train_loader, val_loader, test_loader, train_subset, val_subset, model, device = prepare_data_and_model()

for parametro in model.parameters():
    parametro.requires_grad = False # Congelo pesi

parametri_addestrabili = 0
for parametro in model.parameters():
    if parametro.requires_grad == True:
        parametri_addestrabili += parametro.numel()
print(f"Parametri addestrabili: {parametri_addestrabili}")

features_extractor = model.features.to(device)
features_extractor.eval()

def extract_features(dataloader):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            features = features_extractor(images)

            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.vstack(all_features), np.concatenate(all_labels) # Unione

print("\nInizio estrazione feature e addestramento SVM")
start_time = time.time()

X_train, y_train = extract_features(train_loader)
print(f"Formato feature Train:  {X_train.shape}")

X_test, y_test = extract_features(test_loader)
print(f"Formato feature Test:  {X_test.shape}")

# SVM
svm_classificatore_lineare = SVC(kernel='linear', random_state=42)
svm_classificatore_lineare.fit(X_train, y_train)

svm_classificatore_rbf = SVC(kernel='rbf', random_state=42)
svm_classificatore_rbf.fit(X_train, y_train)

end_time = time.time() 
print(f"Tempo totale (Estrazione + Addestramento 2 SVM): {end_time - start_time:.2f} secondi")


y_pred_lin = svm_classificatore_lineare.predict(X_test)
y_pred_rbf = svm_classificatore_rbf.predict(X_test)

report_lin = classification_report(y_test, y_pred_lin, zero_division=0)
report_rbf = classification_report(y_test, y_pred_rbf, zero_division=0)
print("Report lineare", report_lin)
print("Report rbf", report_rbf)

nomi_classi = train_subset.dataset.classes
plot_confusion_matrix(y_test, y_pred_lin, nomi_classi, titolo_grafico="Approccio 3 (SVM Linear su Features)")
plot_confusion_matrix(y_test, y_pred_rbf, nomi_classi, titolo_grafico="Approccio 3 (SVM RBF su Features)")


# Salvo checkpoint
base_dir = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(base_dir, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)

percorso_file_rbf = os.path.join(checkpoints_dir, 'checkpoint_approach_3_rbf.pkl')
joblib.dump(svm_classificatore_rbf, percorso_file_rbf)
print(f"\nCheckpoint SVM RBF salvato correttamente in: {percorso_file_rbf}")

percorso_file_lin = os.path.join(checkpoints_dir, 'checkpoint_approach_3_linear.pkl')
joblib.dump(svm_classificatore_lineare, percorso_file_lin)
print(f"Checkpoint SVM Lineare salvato correttamente in: {percorso_file_lin}")
