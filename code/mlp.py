import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

NUM_TARGET_CLASSES = 3
RANDOM_STATE = 35
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = "data/processed_data/"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def get_train_data():
    X_path = os.path.join(BASE_PATH, "X_train_scaled.csv")
    y_path = os.path.join(BASE_PATH, "y_train.csv")
    return _get_data_from_csv(X_path, y_path)

def get_test_data():
    X_path = os.path.join(BASE_PATH, "X_test_scaled.csv")
    y_path = os.path.join(BASE_PATH, "y_test.csv")
    return _get_data_from_csv(X_path, y_path)

def _get_data_from_csv(X_path, y_path):
    if not os.path.isfile(X_path) or not os.path.isfile(y_path):
        print("Error: File not found.")
        return None, None
    try:
        X = pd.read_csv(X_path, low_memory=False)
        y = pd.read_csv(y_path, low_memory=False).squeeze()
        return X, y
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse CSV - {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    return None, None

def prepare_datasets(X_train, y_train, X_test, y_test):
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train
    )
    X_train_tensor = torch.tensor(X_train_split.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_split.values, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "X_test_tensor": X_test_tensor,
        "y_test_tensor": y_test_tensor,
    }

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_classes=3, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def regularization_penalty(model, l1_lambda=0.0):
    penalty = 0.0
    if l1_lambda > 0:
        penalty += l1_lambda * sum(param.abs().sum() for param in model.parameters())
    return penalty

def run_epoch(model, loader, criterion, optimizer=None, l1_lambda=0.0):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss = loss + regularization_penalty(model, l1_lambda)
            if training:
                loss.backward()
                optimizer.step()
        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * X_batch.size(0)
        total_correct += (preds == y_batch).sum().item()
        total_samples += y_batch.size(0)
    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc

def train_model(
    model,
    train_loader,
    val_loader,
    learning_rate=0.001,
    num_epochs=100,
    l1_lambda=0.0,
    l2_lambda=0.0,
    early_stopping=False,
    patience=10
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=l2_lambda
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    best_val_loss = float("inf")
    best_state_dict = None
    wait = 0
    model.to(DEVICE)
    for _ in range(num_epochs):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            l1_lambda=l1_lambda
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            l1_lambda=l1_lambda
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if early_stopping and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model, history

def predict_model(model, X_tensor):
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        logits = model(X_tensor.to(DEVICE))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities

def get_model_configs():
    return {
        "Baseline MLP": {
            "dropout_rate": 0.0,
            "l1_lambda": 0.0,
            "l2_lambda": 0.0,
            "early_stopping": False
        },
        "Dropout MLP": {
            "dropout_rate": 0.3,
            "l1_lambda": 0.0,
            "l2_lambda": 0.0,
            "early_stopping": False
        },
        "L1 MLP": {
            "dropout_rate": 0.0,
            "l1_lambda": 1e-5,
            "l2_lambda": 0.0,
            "early_stopping": False
        },
        "L2 MLP": {
            "dropout_rate": 0.0,
            "l1_lambda": 0.0,
            "l2_lambda": 1e-4,
            "early_stopping": False
        },
        "L1 + L2 MLP": {
            "dropout_rate": 0.0,
            "l1_lambda": 1e-5,
            "l2_lambda": 1e-4,
            "early_stopping": False
        },
        "Full Regularized MLP": {
            "dropout_rate": 0.3,
            "l1_lambda": 1e-5,
            "l2_lambda": 1e-4,
            "early_stopping": True
        }
    }

def train_all_models(input_dim, train_loader, val_loader):
    model_configs = get_model_configs()
    models = {}
    histories = {}
    for name, config in model_configs.items():
        model = PyTorchMLP(
            input_dim=input_dim,
            hidden_dim=32,
            num_classes=NUM_TARGET_CLASSES,
            dropout_rate=config["dropout_rate"]
        )
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=LEARNING_RATE,
            num_epochs=NUM_EPOCHS,
            l1_lambda=config["l1_lambda"],
            l2_lambda=config["l2_lambda"],
            early_stopping=config["early_stopping"],
            patience=PATIENCE
        )
        models[name] = trained_model
        histories[name] = history
    return models, histories

def select_best_model(models, histories):
    best_model_name = None
    best_model = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    for name, history in histories.items():
        model_val_acc = max(history["val_acc"])
        model_val_loss = min(history["val_loss"])
        if model_val_acc > best_val_acc or (
            model_val_acc == best_val_acc and model_val_loss < best_val_loss
        ):
            best_val_acc = model_val_acc
            best_val_loss = model_val_loss
            best_model_name = name
            best_model = models[name]
    return best_model_name, best_model

def evaluate_model(model, X_test_tensor, y_test_tensor):
    y_test = y_test_tensor.cpu().numpy().ravel()
    y_pred, _ = predict_model(model, X_test_tensor)
    test_accuracy = np.mean(y_test == y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    cm = pd.crosstab(
        pd.Series(y_test, name="Actual"),
        pd.Series(y_pred, name="Predicted")
    )
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def run_algorithm():
    global NUM_TARGET_CLASSES
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()
    if X_train is None or y_train is None or X_test is None or y_test is None:
        return
    NUM_TARGET_CLASSES = len(np.unique(np.array(y_train).ravel()))
    datasets = prepare_datasets(X_train, y_train, X_test, y_test)
    input_dim = X_train.shape[1]
    models, histories = train_all_models(
        input_dim=input_dim,
        train_loader=datasets["train_loader"],
        val_loader=datasets["val_loader"]
    )
    _, best_model = select_best_model(models, histories)
    evaluate_model(
        model=best_model,
        X_test_tensor=datasets["X_test_tensor"],
        y_test_tensor=datasets["y_test_tensor"]
    )

def main():
    run_algorithm()

if __name__ == "__main__":
    main()
