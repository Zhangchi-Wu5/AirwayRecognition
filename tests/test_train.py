"""Tests for training utilities (set_seed, train_epoch, validate_epoch)."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.train import set_seed, train_one_epoch, validate


def make_tiny_loader(n_samples: int = 16, batch_size: int = 4):
    x = torch.randn(n_samples, 3, 224, 224)
    y = torch.randint(0, 3, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def make_tiny_model():
    # Tiny CNN that accepts (3, 224, 224) and outputs 3 classes
    return torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(3, 3),
    )


def test_set_seed_reproducible():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_train_one_epoch_returns_loss_and_acc():
    model = make_tiny_model()
    loader = make_tiny_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = train_one_epoch(model, loader, optimizer, criterion, device="cpu")
    assert 0.0 <= loss < 100.0
    assert 0.0 <= acc <= 1.0


def test_validate_returns_loss_and_acc():
    model = make_tiny_model()
    loader = make_tiny_loader()
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = validate(model, loader, criterion, device="cpu")
    assert 0.0 <= loss < 100.0
    assert 0.0 <= acc <= 1.0


def test_train_reduces_loss_on_memorizable_data():
    """Sanity check: training for a few epochs on tiny data should reduce loss."""
    set_seed(42)
    model = make_tiny_model()
    loader = make_tiny_loader(n_samples=8, batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    initial_loss, _ = validate(model, loader, criterion, device="cpu")
    for _ in range(5):
        train_one_epoch(model, loader, optimizer, criterion, device="cpu")
    final_loss, _ = validate(model, loader, criterion, device="cpu")
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} → {final_loss}"
