import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from autoencoder import Autoencoder
import matplotlib.pyplot as plt
from dataio import tennessee_dataset


def train(train_data_path, test_data_path, input_dim, latent_dim, reconstruct_dim):
    
    ds = tennessee_dataset(train_data_path, test_data_path)

    # Create DataLoader
    batch_size = 16
    train_dataset = ds.get_train_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test = ds.get_test_dataset()
    test_loader = DataLoader(test, batch_size=1, shuffle=False)


    autoencoder = Autoencoder(input_dim, latent_dim, reconstruct_dim)

    # Step 3: Train the autoencoder using the training data
    criterion = nn.MSELoss()
    optimizer = optim.SGD(autoencoder.parameters(), lr=0.1, momentum=0.9)

    model_loss = []

    num_epochs = 10
    for epoch in range(num_epochs):
        for data in train_loader:
            inputs = data[0] # Access the first element of the tuple, which contains the inputs
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            model_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    plt.plot(model_loss)
    plt.show()

    # Use train data again to tune the threshold
    train_loader_tuning = DataLoader(train_dataset, batch_size=1, shuffle=True)

    autoencoder.eval()  # Set the model to evaluation mode (no gradient computation)
    train_losses_tuning = []
    with torch.no_grad():
        for data in train_loader_tuning:
            inputs = data[0] # Access the first element of the tuple, which contains the inputs
            outputs = autoencoder(inputs)
            train_loss_tuning = criterion(outputs, inputs)
            train_losses_tuning.append(train_loss_tuning.item())

    plt.plot(train_losses_tuning)
    plt.show()



if __name__ == '__main__':
    train_path = 'https://raw.githubusercontent.com/iraola/te-orig-fortran/main/datasets/braatz_anomaly_detection/train/d00.csv'

    test_path = 'https://raw.githubusercontent.com/iraola/te-orig-fortran/main/datasets/braatz_anomaly_detection/test/d01.csv'

    train(train_path, test_path, 52, [32], [32])