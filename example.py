import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  # Import tqdm for progress bar

from model import LanguageModel  # Import the complete model


def generate_random_data(num_samples, seq_len, input_dim, output_dim):
    """
    Generate random sequences of input vectors and corresponding target sequences.

    Args:
        num_samples (int): Number of samples to generate.
        seq_len (int): Sequence length.
        input_dim (int): Dimension of the input vectors.
        output_dim (int): Dimension of the output vectors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input and target sequences.
    """
    x = torch.rand(num_samples, seq_len, input_dim)
    y = torch.randint(0, output_dim, (num_samples, seq_len))
    return x, y


def train_model(model, loss_fn, optimizer, x_train, y_train, x_val, y_val, epochs, batch_size, device):
    """
    Train the model on the training data and evaluate on validation data.

    Args:
        model (nn.Module): The model to train.
        loss_fn (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for training.
        x_train (Tensor): Training inputs.
        y_train (Tensor): Training targets.
        x_val (Tensor): Validation inputs.
        y_val (Tensor): Validation targets.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and validation.
    """
    # Prepare data loaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print("Data loaders created and starting training")
    for epoch in range(epochs):
        print(f"Starting with {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits, aux_loss = model(x_batch)
            # Reshape for loss computation
            logits = logits.view(-1, logits.size(-1))
            y_batch = y_batch.view(-1)
            loss = loss_fn(logits, y_batch)
            total_loss = loss + aux_loss
            total_loss.backward()
            optimizer.step()

            # Calculate training accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_train_loss = total_loss.item() / len(train_loader)
        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits, _ = model(x_batch)
                logits = logits.view(-1, logits.size(-1))
                y_batch = y_batch.view(-1)
                val_loss = loss_fn(logits, y_batch)
                total_val_loss += val_loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


def main():
    """
    Main function to set up data, model, and start training.
    """
    # Hyperparameters
    input_dim = 1457
    hidden_dim = 512
    output_dim = 10
    num_experts = 10
    seq_len = 2
    batch_size = 64
    k = 2
    epochs = 5  # Adjust as needed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Generate random data
    num_samples = 10000
    x_data, y_data = generate_random_data(num_samples, seq_len, input_dim, output_dim)

    # Split data into training and validation sets
    split_idx = int(0.8 * num_samples)
    x_train, x_val = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]

    # Instantiate the LanguageModel
    model = LanguageModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                          num_experts=num_experts, k=k)
    model = model.to(device)
    print(f"Model created {model}")

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    # Move data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    print("Data moved to device")
    # Train the model
    train_model(model, loss_fn, optimizer, x_train, y_train, x_val, y_val, epochs, batch_size, device)


if __name__ == '__main__':
    main()
