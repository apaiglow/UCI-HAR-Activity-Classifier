import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import pickle
from sklearn.metrics import accuracy_score, classification_report

from preprocess import load_and_preprocess
from model import MyNN


X_train_pca, X_test_pca, y_train, y_test = load_and_preprocess()


class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.squeeze(), dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


train_dataset = CustomDataset(X_train_pca, y_train)
test_dataset = CustomDataset(X_test_pca, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)


class_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS',
               'SITTING', 'STANDING', 'LAYING']


def objective(trial):

    num_hidden_layer = trial.suggest_int("num_hidden_layers", 1, 5)
    neurons_per_layer = trial.suggest_int("neurons_per_layers", 8, 128, step=8)
    dropout_ratio = trial.suggest_float("dropout_ratio", 0.1, 0.5)

    input_dim = X_train_pca.shape[1]
    output_dim = len(class_names)

    model = MyNN(input_dim, output_dim, num_hidden_layer, neurons_per_layer, dropout_ratio)

    epochs = 100
    learning_rate = 0.1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(epochs):
        for batch_features, batch_labels in train_loader:

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:

            outputs = model(batch_features)
            predicted = outputs.argmax(dim=1)

            total += batch_labels.shape[0]
            correct += (predicted == batch_labels).sum().item()

    return correct / total


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params


model = MyNN(
    X_train_pca.shape[1],
    len(class_names),
    best_params['num_hidden_layers'],
    best_params['neurons_per_layers'],
    best_params['dropout_ratio']
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

for epoch in range(100):
    for batch_features, batch_labels in train_loader:

        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


y_pred = []
y_true = []

model.eval()

with torch.no_grad():
    for batch_features, batch_labels in test_loader:

        outputs = model(batch_features)
        predicted = outputs.argmax(dim=1)

        y_pred.extend(predicted.numpy())
        y_true.extend(batch_labels.numpy())


print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=class_names))


with open("../models/best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)

with open("../models/class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)

torch.save(model.state_dict(), "../models/ann_model.pth")