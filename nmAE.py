import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from nets.AutoEncoder import nmODEAutoencoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import datetime
from thop import profile

epoches = [1]  # [10, 20]
input_size = 784  # MNIST images are 28x28 in size and are expanded to 784-dimensional vectors.
hidden_size = 500
batch_size = 128
eval_time = (0, 0.8)
noising_factors = [0, 0.15, 0.4, 0.6]
total_train_time = 0


# 添加高斯噪声
def add_gaussian_noise(image, noise_factor):
    noise = noise_factor * torch.randn(image.size())
    noisy_image = image + noise
    # clip
    noisy_image = np.clip(noisy_image, 0., 1.)
    return noisy_image


def hidden_Knn_test(eval_time):
    test_noising_factor = 0
    checkpoint = torch.load(f'./ae_dict.pkl')
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: add_gaussian_noise(x, noise_factor)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data/test', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    opt_model = nmODEAutoencoder(input_size, hidden_size, eval_times=eval_time)
    opt_model.load_state_dict(checkpoint)
    opt_model.eval()

    hidden_features = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            _, hidden = opt_model(images)
            hidden_features.append(hidden)
    hidden_features = torch.cat(hidden_features, dim=0)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: add_gaussian_noise(x, test_noising_factor)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data/train', train=True, download=False, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    train_hidden_features = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.view(images.size(0), -1)
            _, hidden = opt_model(images)
            train_hidden_features.append(hidden)
    train_hidden_features = torch.cat(train_hidden_features, dim=0)
    test_hidden_features = hidden_features
    X_train = train_hidden_features.numpy()
    y_train = train_dataset.targets.numpy()
    X_test = test_hidden_features.numpy()
    y_test = test_dataset.targets.numpy()

    # Using hidden features for KNN classification
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Accuracy: {:.2%}".format(accuracy))
    return accuracy


model_select = "nmODE"
for epoch_num in epoches:
    print(f"------------------{model_select}---------Epoch{epoch_num}----noise_factor:{noise_factor}----")
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    noise_factor = noise_factor
    learning_rate = 0.0005
    num_epochs = epoch_num
    # load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data/train', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    transform_noisy = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: add_gaussian_noise(x, noise_factor)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset_noisy = datasets.MNIST(root='./data/train', train=True, download=False,
                                         transform=transform_noisy)

    train_loader_noisy = DataLoader(train_dataset_noisy, batch_size=batch_size, shuffle=False)
    test_dataset = datasets.MNIST(root='./data/test', train=False, download=False, transform=transform_noisy)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    model = nmODEAutoencoder(input_size, hidden_size, eval_times=eval_time, adjoint=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        starttime = datetime.datetime.now()
        total_loss = 0
        total_sparsity_penalty = 0

        for batch_images, batch_images_noisy in zip(train_loader, train_loader_noisy):
            batch_images = batch_images[0].to(device)
            batch_images_noisy = batch_images_noisy[0].to(device)
            batch_images = batch_images.view(-1, input_size)
            batch_images_noisy = batch_images_noisy.view(-1, input_size)
            reconstructions, hidden = model(batch_images_noisy)
            flops, params = profile(model, (batch_images_noisy,))
            print('flops: ', flops, 'params: ', params)
            reconstruction_loss = nn.MSELoss()(reconstructions, batch_images)
            loss = reconstruction_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        endtime = datetime.datetime.now()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        # Display the image before, after and after decoding
        acc = hidden_Knn_test(eval_time)
        torch.save(model.state_dict(),
                   f"./ae_dict.pkl")

    with torch.no_grad():
        batch_times = 0
        for batch_images, _ in test_loader:
            batch_images = batch_images.view(-1, input_size).to(device)
            reconstructions, hidden = model(batch_images)

            batch_images = batch_images.view(-1, 28, 28)
            reconstructions = reconstructions.view(-1, 28, 28)

            for i in range(10):
                batch_images_cpu = batch_images.cpu()
                plt.subplot(3, 10, i + 1)
                plt.imshow(batch_images_cpu[i], cmap='gray')
                plt.axis('off')
                
                reconstructions_cpu = reconstructions.cpu()
                plt.subplot(3, 10, i + 11)
                plt.imshow(reconstructions_cpu[i], cmap='gray')
                plt.axis('off')

            plt.show()
            batch_times = batch_times + 1
            if batch_times > 2:
                break
