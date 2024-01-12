import click  # Import the click library for creating command-line interfaces
import torch  # Import the PyTorch library for deep learning capabilities
from torch import nn, optim  # Import neural network modules from PyTorch
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from model import MyAwesomeModel  # Import the custom neural network model
from data import mnist  # Import the function to load MNIST data
import wandb

@click.group()
def cli():
    """
    Command line interface.

    This function defines the main command-line interface using the click library.
    It serves as a grouping mechanism for multiple commands.
    """
    pass  # Placeholder for an empty function, as required by click

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """
    Train a model on MNIST.

    Args:
        lr (float): Learning rate for model training.
    """
    print("Training day and night")
    print(f"Learning Rate: {lr}")

    # Initialize wandb
    wandb.init(project="logging_experiment", name="experiment1")  # Replace with your project and experiment names

    # Get the training set
    train_set, _ = mnist()
    
    # Initialize the model
    model = MyAwesomeModel()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_losses = []
    for epoch in range(30):
        running_loss = 0.0
        for images, labels in train_set:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_set))

        # Log training loss to wandb
        wandb.log({"Training Loss": running_loss / len(train_set)})

    # Save the trained model
    torch.save(model, "trained_model.pt")
    
    # Plot training curve
    plt.plot(train_losses)
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.title('Training Curve')
    plt.show()

    # Log example image to wandb
    example_image, _ = next(iter(train_set))
    wandb.log({"Image": [wandb.Image(example_image[0], caption="Example Image")]})

    # Log histogram to wandb
    random_data = torch.randn(1000)
    wandb.log({"Data Histogram": wandb.Histogram(random_data.numpy())})
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """
    Evaluate a trained model.

    Args:
        model_checkpoint (str): Path to the saved model checkpoint.

    This command evaluates a pre-trained model on the test set of the MNIST dataset.
    Let's understand some terms:

    1. `torch.load(model_checkpoint)`: Loads a pre-trained model from the specified checkpoint file.
       In PyTorch, model checkpoints often contain saved model weights and architecture.

    2. Evaluation loop: It involves making predictions on the test set and calculating accuracy.
       Accuracy is the ratio of correct predictions to the total number of predictions.

    """
    print("Evaluating like my life depends on it")
    print(f"Model Checkpoint: {model_checkpoint}")  # Print the provided model checkpoint path

    # Get the test set
    _, test_set = mnist()

    # Load the trained model
    model = torch.load(model_checkpoint)
    
    # Evaluation loop
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_set:
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = correct / total
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
