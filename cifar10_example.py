import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import LanguageModel  # Use the same model as in example.py


def train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs, device):
    """
    Train the model on the training data and evaluate on validation data.
    """
    for epoch in range(epochs):
        print(f"Starting with {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, aux_loss = model(x_batch)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1)) + aux_loss
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            running_loss += loss.item() * x_batch.size(0)

            # Calculate training accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_train_loss = running_loss / total
        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits, aux_loss = model(x_batch)
                val_loss = loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1)) + aux_loss
                total_val_loss += val_loss.item() * x_batch.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        avg_val_loss = total_val_loss / total
        val_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


def inspect_experts(model):
    """
    Inspect the experts in the MoE layer after training.
    """
    moe_layer = model.moe
    for idx, expert in enumerate(moe_layer.experts):
        print(f"Expert {idx} parameters:")
        for name, param in expert.named_parameters():
            print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")


def main():
    """
    Main function to set up data, model, and start training.
    """
    # Hyperparameters
    input_dim = 3 * 32 * 32  # CIFAR-10 images are 32x32 RGB images
    hidden_dim = 512
    output_dim = 10  # CIFAR-10 has 10 classes
    num_experts = 10
    batch_size = 64
    k = 4
    epochs = 5  # Adjust as needed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Prepare CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the LanguageModel
    model = LanguageModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                          num_experts=num_experts, k=k)
    model = model.to(device)
    print(f"Model created {model}")

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    # Train the model
    train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs, device)

    # Inspect the experts after training
    inspect_experts(model)


if __name__ == '__main__':
    main()

