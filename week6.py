"""
MNIST Handwritten Digits Classification Assignment

This script implements neural network models for classifying 
MNIST handwritten digits, addressing the following problems:
1. Dataset Loading and Visualization
2. Single Hidden Layer Neural Network
3. Two Hidden Layers with L2 Regularization
4. Convolutional Neural Network

Requirements:
- Use PyTorch for model implementation
- Explore different network architectures
- Report validation accuracy for each epoch
"""

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F

class DetailedModelEvaluator:
    @staticmethod
    def calculate_accuracy(model, data_loader, device="cpu"):
        """
        Calculate model accuracy on the given data loader.
        
        Args:
            model (nn.Module): Neural network model to evaluate
            data_loader (DataLoader): Dataset to evaluate on
            device (str): Device to run the evaluation on
        
        Returns:
            float: Accuracy of the model
        """
        model.eval()
        total_correct = 0
        total_instances = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Improved prediction handling
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total_correct += (predicted == labels).sum().item()
                total_instances += labels.size(0)
        
        return total_correct / total_instances

    @staticmethod
    def train_model(model, data_loader, optimizer, loss_function, device="cpu"):
        """
        Train the model for one epoch.
        
        Args:
            model (nn.Module): Neural network model to train
            data_loader (DataLoader): Training dataset
            optimizer (torch.optim): Optimization algorithm
            loss_function (nn.Module): Loss calculation method
            device (str): Device to run the training on
        """
        model.train()
        total_loss = 0.0
        
        for data, target in data_loader:
            # Move data to specified device
            data = data.to(device)
            target = target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = loss_function(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)

def visualization_check(train_data, test_data):
    """
    Visualize and verify MNIST dataset characteristics.
    
    Args:
        train_data (MNIST): Training dataset
        test_data (MNIST): Test dataset
    """
    plt.figure(figsize=(12, 5))
    
    # Training data visualization
    plt.subplot(1, 2, 1)
    train_image, train_label = train_data[0]
    plt.title(f"Train Sample (Label: {train_label})")
    plt.imshow(train_image.squeeze(), cmap='gray')
    print(f"Training Image Shape: {train_image.shape}")
    print(f"Training Image Value Range: [{train_image.min()}, {train_image.max()}]")
    
    # Test data visualization
    plt.subplot(1, 2, 2)
    test_image, test_label = test_data[0]
    plt.title(f"Test Sample (Label: {test_label})")
    plt.imshow(test_image.squeeze(), cmap='gray')
    print(f"Test Image Shape: {test_image.shape}")
    print(f"Test Image Value Range: [{test_image.min()}, {test_image.max()}]")
    
    plt.tight_layout()
    plt.show()

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST datasets
    train_data = MNIST("./", train=True, download=True, transform=ToTensor())
    test_data = MNIST("./", train=False, download=True, transform=ToTensor())
    
    # Visualization and dataset verification
    visualization_check(train_data, test_data)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
    
    # Loss function
    loss_function = nn.CrossEntropyLoss()
    
    # Problem 2: Single Hidden Layer Network
    print("\n--- Problem 2: Single Hidden Layer Network ---")
    single_layer_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 300),
        nn.ReLU(),
        nn.Linear(300, 10)
    ).to(device)
    
    optimizer = torch.optim.SGD(single_layer_model.parameters(), lr=0.1, momentum=0.9)
    
    print("Epoch-wise Accuracy:")
    for epoch in range(10):
        train_loss = DetailedModelEvaluator.train_model(single_layer_model, train_loader, optimizer, loss_function, device)
        accuracy = DetailedModelEvaluator.calculate_accuracy(single_layer_model, test_loader, device)
        print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.4f}, Train Loss = {train_loss:.4f}")
    
    # Problem 3: Two Hidden Layers with L2 Regularization
    print("\n--- Problem 3: Two Hidden Layers with L2 Regularization ---")
    two_layer_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 500),
        nn.ReLU(),
        nn.Linear(500, 300),
        nn.ReLU(),
        nn.Linear(300, 10)
    ).to(device)
    
    optimizer = torch.optim.SGD(two_layer_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    
    print("Epoch-wise Accuracy:")
    for epoch in range(40):
        train_loss = DetailedModelEvaluator.train_model(two_layer_model, train_loader, optimizer, loss_function, device)
        accuracy = DetailedModelEvaluator.calculate_accuracy(two_layer_model, test_loader, device)
        print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.4f}, Train Loss = {train_loss:.4f}")
    
    # Problem 4: Convolutional Neural Network
    print("\n--- Problem 4: Convolutional Neural Network ---")
    cnn_model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 14 * 14, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)
    
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    
    print("Epoch-wise Accuracy:")
    for epoch in range(40):
        train_loss = DetailedModelEvaluator.train_model(cnn_model, train_loader, optimizer, loss_function, device)
        accuracy = DetailedModelEvaluator.calculate_accuracy(cnn_model, test_loader, device)
        print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.4f}, Train Loss = {train_loss:.4f}")

if __name__ == "__main__":
    main()
"""
Key Modifications and Improvements:
1. Added device support (CPU/CUDA)
2. Improved visualization with shape and value range checks
3. More detailed training loop with loss tracking
4. Removed Sigmoid activation (not appropriate for classification)
5. Added momentum to SGD optimizer
6. Corrected CNN architecture for proper dimensionality
7. More comprehensive documentation

Note: Actual performance may vary due to randomness in initialization.
Recommended to run multiple times to verify consistency.
"""