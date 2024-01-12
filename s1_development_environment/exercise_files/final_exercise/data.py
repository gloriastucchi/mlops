import torch  # Import the PyTorch library for deep learning capabilities
from torch.utils.data import DataLoader  # Import the DataLoader class for efficient data loading

def mnist(data_folder='C:\\Users\\stucc\\OneDrive\\Desktop\\mlops\\CookieCutterMLOps\\data\\raw\\corruptmnist', batch_size=64, num_workers=0):
    """
    Load and prepare training and test data for the Corrupted MNIST dataset using PyTorch's DataLoader.

    Args:
        data_folder (str): The path to the directory containing Corrupted MNIST data.
        batch_size (int): The batch size for DataLoader, specifying the number of samples per batch.
        num_workers (int): The number of subprocesses to use for data loading. Set to 0 for sequential loading.

    Returns:
        tuple: Tuple containing DataLoader instances for training and test datasets.
    """
    
    # Load train data
    # Concatenate training images from multiple files (0 to 5) along the 0th dimension
    train_images = torch.cat([torch.load(f'{data_folder}/train_images_{i}.pt') for i in range(6)], dim=0)
    # Concatenate corresponding training targets along the 0th dimension
    train_targets = torch.cat([torch.load(f'{data_folder}/train_target_{i}.pt') for i in range(6)], dim=0)
    
    # Combine images and targets into a single TensorDataset
    train_dataset = torch.utils.data.TensorDataset(train_images, train_targets)
    
    # Create a DataLoader for training data
    # Batches, shuffles, and uses multiple workers for parallel loading
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Load test data
    # Load test images and targets
    test_images = torch.load(f'{data_folder}/test_images.pt')
    test_targets = torch.load(f'{data_folder}/test_target.pt')
    
    # Combine test images and targets into a single TensorDataset
    test_dataset = torch.utils.data.TensorDataset(test_images, test_targets)
    
    # Create a DataLoader for test data
    # Batches and uses multiple workers for parallel loading
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader  # Return DataLoader instances for training and test datasets
