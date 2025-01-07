import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def fit(
    model, x_train, x_test, y_train, y_test, loss_fn, epochs=100, mode="classification"
):
    """fit
    Fits a given model to a given dataset for a specified number of epochs.
    """
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            # targets = torch.argmax(outputs, dim = 1)
            # print(outputs.shape, outputs.dtype)
            # print(labels.shape, labels.dtype)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze()
                loss = loss_fn(outputs, labels)
                if mode == "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            if mode == "classification":
                print(f"Epoch {i+1}/{epochs}, Accuracy: {correct/total}")


def fit_v2(
    model, x_train, x_test, y_train, y_test, loss_fn, epochs=100, mode="classification"
):
    """
    Trains the model using the given data and loss function.

    Args:
        model: PyTorch model to be trained.
        x_train: Training features (torch.Tensor).
        x_test: Testing features (torch.Tensor).
        y_train: Training labels (torch.Tensor).
        y_test: Testing labels (torch.Tensor).
        loss_fn: Loss function to be used for training.
        epochs: Number of epochs for training (default: 100).
        mode: Either "classification" or "regression" (default: "classification").
    """
    # Ensure the model is on the correct device
    model = model.to(next(model.parameters()).device)

    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Optimizer (Adam by default)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = []

        for itr, (batch_x, batch_y) in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_x)

            # Compute loss
            loss = loss_fn(predictions, batch_y)
            # print(f"Batch {itr+1}, Loss: {loss.item()}")

            # Backward pass
            loss.backward()

            # Optimization step
            optimizer.step()

            # Accumulate running loss
            running_loss.append(loss.item())

            if itr == 25:
                for g in optimizer.param_groups:
                    g["lr"] = 0.0001

            if itr == 100:
                for g in optimizer.param_groups:
                    g["lr"] = 0.00001

        epoch_loss = np.average(running_loss)
        # Evaluation on test data
        model.eval()
        with torch.no_grad():
            test_predictions = model(x_test)
            if mode == "classification":
                test_loss = loss_fn(test_predictions, y_test).item()
                accuracy = (
                    (test_predictions.argmax(dim=1) == y_test).float().mean().item()
                )
                print(
                    f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
                )
            elif mode == "regression":
                test_loss = loss_fn(test_predictions.squeeze(), y_test).item()
                print(
                    f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}"
                )
