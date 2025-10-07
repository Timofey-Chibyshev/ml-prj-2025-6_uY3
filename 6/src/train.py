import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from save import save_model

def train_model(model, criterion, optimizer, x_train, y_train, batch_size=100, num_epochs=5):
    x_train_normalized = np.array([np.array([x_train_elem]) for x_train_elem in x_train])

    x_train_tensor = torch.Tensor(x_train_normalized)
    y_train_tensor = torch.LongTensor(y_train)

    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Прямой запуск
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Отслеживание точности
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                            (correct / total) * 100))

if __name__ == "__main__":
    import sys
    from model import MnistNumbersCNN
    from data_loader import loader
    from torch import nn, optim

    print("Loading data...")
    x_train, y_train = loader.load_train_data()
    print("Training data has been loaded successfully!\n")

    learning_rate = 0.001

    model = MnistNumbersCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print("Training model...")
    train_model(model, criterion, optimizer, x_train, y_train, num_epochs=3)
    print("Model has been trained successfully!\n")

    filename = "mnist_numbers_cnn"
    save_model(model, f"{filename}.ckpt")
    print(f"Model saved to \"{filename}.ckpt\"")
